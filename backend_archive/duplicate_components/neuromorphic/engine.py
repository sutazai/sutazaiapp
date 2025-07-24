import grpc
from concurrent import futures
import neuromorphic_pb2
import neuromorphic_pb2_grpc
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import MSTDP, STDP, Hebbian
from bindsnet.encoding import poisson
import numpy as np
import time
import logging
import torch
import os
import psutil
from typing import Optional
import nengo
from nengo.connection import Connection
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/neuromorphic.log"), logging.StreamHandler()],
)
logger = logging.getLogger("NeuromorphicEngine")

# Define time constant
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class EnergyMonitor:
    """Tracks energy usage of neuromorphic computations"""

    def __init__(self):
        self.baseline_power = self._measure_baseline()
        self.accumulated_energy = 0.0  # Joules
        self.last_measured = time.time()
        logger.info(f"Baseline power consumption: {self.baseline_power:.4f} W")

    def _measure_baseline(self) -> float:
        """Measure baseline power consumption"""
        # This is an approximation - in a real system we'd use hardware power monitors
        # Estimate using CPU utilization as proxy
        measurements = []
        for _ in range(5):
            measurements.append(psutil.cpu_percent(interval=0.2))
        # Convert CPU % to watts (very rough approximation)
        # Assuming 1% CPU = 0.3W on our server
        return float(np.mean(measurements) * 0.3)

    def start_measurement(self):
        """Start energy measurement period"""
        self.last_measured = time.time()

    def end_measurement(self) -> float:
        """End measurement and return energy used in joules"""
        duration = time.time() - self.last_measured
        current_power = psutil.cpu_percent() * 0.3  # Watts
        # Energy = power * time
        energy_used = (current_power - self.baseline_power) * duration
        self.accumulated_energy += max(0, energy_used)  # Ensure non-negative
        return energy_used

    def get_total_energy(self) -> float:
        """Get total accumulated energy in joules"""
        return self.accumulated_energy


class NeuromorphicEngineServicer(neuromorphic_pb2_grpc.NeuromorphicEngineServicer):
    def __init__(self):
        """Initialize the neuromorphic engine with default SNN configuration"""
        self.network = Network(dt=1.0)
        self.energy_monitor = EnergyMonitor()
        self._init_base_network()
        self.resource_usage = {
            "energy_joules": 0.0,
            "compute_seconds": 0.0,
            "spike_ops": 0,
            "memory_bytes": 0,
        }
        self.plasticity_enabled = True
        self.input_layer: Optional[Input] = None
        self.excitatory_layer: Optional[LIFNodes] = None
        self.inhibitory_layer: Optional[LIFNodes] = None
        self.output_layer: Optional[IzhikevichNodes] = None

        # Connections
        self.input_exc_conn: Optional[Connection] = None
        self.exc_inh_conn: Optional[Connection] = None
        self.inh_exc_conn: Optional[Connection] = None
        self.exc_out_conn: Optional[Connection] = None

        logger.info("Neuromorphic engine initialized")

    def _init_base_network(self):
        """Initialize the spiking neural network architecture"""
        # Create input and output layers
        self.input_layer = Input(n=128, traces=True)
        self.excitatory_layer = LIFNodes(
            n=64, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5
        )
        self.inhibitory_layer = LIFNodes(
            n=64, traces=True, rest=-60.0, reset=-45.0, thresh=-40.0, refrac=2
        )
        self.output_layer = IzhikevichNodes(
            n=10,
            traces=True,
            a=0.02,
            b=0.2,
            c=-65.0,
            d=8.0,
            rest=-65.0,
            reset=-65.0,
            thresh=-40.0,
        )

        # Add layers to network
        self.network.add_layer(self.input_layer, name="input")
        self.network.add_layer(self.excitatory_layer, name="excitatory")
        self.network.add_layer(self.inhibitory_layer, name="inhibitory")
        self.network.add_layer(self.output_layer, name="output")

        # Create connections with plastic synapses
        # Input to excitatory with STDP learning
        self.input_exc_conn = Connection(
            source=self.input_layer,
            target=self.excitatory_layer,
            update_rule=STDP,
            nu=(1e-4, 1e-2),
            wmin=0.0,
            wmax=1.0,
        )

        # Excitatory to inhibitory with Hebbian learning
        self.exc_inh_conn = Connection(
            source=self.excitatory_layer,
            target=self.inhibitory_layer,
            update_rule=Hebbian,
            nu=1e-2,
        )

        # Inhibitory to excitatory (lateral inhibition)
        self.inh_exc_conn = Connection(
            source=self.inhibitory_layer,
            target=self.excitatory_layer,
            w=-0.1 * torch.ones(self.inhibitory_layer.n, self.excitatory_layer.n),
        )

        # Excitatory to output with MSTDP (modulated STDP)
        self.exc_out_conn = Connection(
            source=self.excitatory_layer,
            target=self.output_layer,
            update_rule=MSTDP,
            nu=(1e-4, 1e-2),
            wmin=0.0,
            wmax=1.0,
        )

        # Register all connections
        self.network.add_connection(
            self.input_exc_conn, source="input", target="excitatory"
        )
        self.network.add_connection(
            self.exc_inh_conn, source="excitatory", target="inhibitory"
        )
        self.network.add_connection(
            self.inh_exc_conn, source="inhibitory", target="excitatory"
        )
        self.network.add_connection(
            self.exc_out_conn, source="excitatory", target="output"
        )

        logger.info(
            "Base neuromorphic network initialized with 4 layers and plastic connections"
        )

    def ProcessSpikes(self, request, context):
        """Process incoming spike train and return network activations"""
        self.energy_monitor.start_measurement()
        start_time = time.time()

        try:
            # Decode input spikes from protobuf message
            spike_data = np.frombuffer(request.encoded_spikes, dtype=np.float32)
            spike_data = spike_data.reshape(
                -1, 128
            )  # Reshape to match input layer size

            # Convert to Poisson spike train if needed
            if request.neuron_profile == "poisson":
                spike_data = poisson(
                    torch.from_numpy(spike_data), time=request.temporal_window
                )
            else:
                spike_data = torch.from_numpy(spike_data)

            # Run the network
            inputs = {"input": spike_data}
            self.network.run(inputs=inputs, time=request.temporal_window)

            # Get output spikes and membrane potentials
            output_spikes = self.network.monitors["output"].get("s")
            # F841: Removed unused variable output_voltages
            # output_voltages = self.network.monitors["output"].get("v")

            # Calculate activations (normalized firing rates)
            if output_spikes is None:
                spike_counts = torch.zeros(10)
            else:
                spike_counts = output_spikes.sum(0).float()
            activations = spike_counts / request.temporal_window

            # Update resource tracking
            end_time = time.time()
            duration = end_time - start_time
            energy_used = self.energy_monitor.end_measurement()
            spike_ops = int(self.network.n_neurons * request.temporal_window)

            self.resource_usage["compute_seconds"] += duration
            self.resource_usage["energy_joules"] += energy_used
            self.resource_usage["spike_ops"] += spike_ops
            self.resource_usage["memory_bytes"] = (
                psutil.Process(os.getpid()).memory_info().rss
            )

            # Create response with activations and resource usage
            response = neuromorphic_pb2.SpikeOutput()
            response.activations.extend(activations.numpy().tolist())

            # Add energy usage information
            response.energy_usage[1] = energy_used
            response.processing_metadata = (
                f"compute_time={duration:.4f}s;spike_ops={spike_ops}"
            )

            logger.info(
                f"Processed {spike_ops} spike operations in {duration:.4f}s using {energy_used:.4f}J"
            )
            return response

        except Exception as e:
            logger.error(f"Error processing spikes: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing spikes: {str(e)}")
            return neuromorphic_pb2.SpikeOutput()

    def TrainModel(self, request, context):
        """Train the spiking neural network using the provided spike data"""
        if not self.plasticity_enabled:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Network plasticity is disabled")
            return neuromorphic_pb2.TrainingResult()

        self.energy_monitor.start_measurement()
        start_time = time.time()

        try:
            # Extract training samples
            num_samples = len(request.samples)
            total_loss = 0.0
            total_accuracy = 0.0

            for i, sample in enumerate(request.samples):
                # Decode input spikes
                spike_data = np.frombuffer(sample.encoded_spikes, dtype=np.float32)
                spike_data = spike_data.reshape(-1, 128)

                # Get target output if available
                target = None
                if i < len(request.targets):
                    target = torch.tensor(request.targets[i].activations)

                # Run network with plasticity enabled
                inputs = {"input": torch.from_numpy(spike_data)}
                self.network.run(inputs=inputs, time=sample.temporal_window)

                # Compute loss if target provided
                if target is not None:
                    output = (
                        self.network.monitors["output"].get("s").sum(0).float()
                        / sample.temporal_window
                    )
                    # Simple MSE loss
                    loss = torch.sum((output - target) ** 2)
                    total_loss += loss.item()

                    # Binary accuracy
                    predicted = torch.argmax(output)
                    true_label = torch.argmax(target)
                    total_accuracy += (predicted == true_label).float().item()

                # Reset network state for next sample
                self.network.reset_state_variables()

            # Calculate average metrics
            avg_loss = total_loss / num_samples if num_samples > 0 else 0
            avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0

            # Update resource tracking
            end_time = time.time()
            duration = end_time - start_time
            energy_used = self.energy_monitor.end_measurement()

            self.resource_usage["compute_seconds"] += duration
            self.resource_usage["energy_joules"] += energy_used

            # Create response
            response = neuromorphic_pb2.TrainingResult(
                loss=avg_loss,
                accuracy=avg_accuracy,
                resources_used=neuromorphic_pb2.NeuroResourceUsage(
                    energy_joules=energy_used,
                    compute_seconds=duration,
                    spike_ops=int(
                        self.network.n_neurons
                        * sum(s.temporal_window for s in request.samples)
                    ),
                    memory_bytes=psutil.Process(os.getpid()).memory_info().rss,
                ),
            )

            logger.info(
                f"Trained on {num_samples} samples. Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
            )
            return response

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error training model: {str(e)}")
            return neuromorphic_pb2.TrainingResult()

    def GetResourceUsage(self, request, context):
        """Return resource usage statistics for the neuromorphic engine"""
        # Get current memory usage
        memory_usage = psutil.Process(os.getpid()).memory_info().rss

        return neuromorphic_pb2.NeuroResourceUsage(
            energy_joules=self.resource_usage["energy_joules"],
            compute_seconds=self.resource_usage["compute_seconds"],
            spike_ops=self.resource_usage["spike_ops"],
            memory_bytes=memory_usage,
        )

    def get_bias_current(self, layer_name: str) -> float:
        """Get the bias current for a layer."""
        # This requires access to the underlying BindsNET network object and layers
        # Placeholder implementation:
        try:
            if layer_name in self.network.layers:
                # Assuming bias current might be related to layer threshold or similar
                # This is highly dependent on the BindsNET layer implementation
                # Example: access threshold if available
                if hasattr(self.network.layers[layer_name], 'thresh'):
                     # Return the threshold value, cast to float
                     return float(self.network.layers[layer_name].thresh)
                else:
                     return 0.0 # No relevant attribute found
            else:
                return 0.0
        except Exception as e:
             logger.warning(f"Could not retrieve bias current for {layer_name}: {e}")
             # Ensure float return on error path too
             return 0.0

    def apply_plasticity(self, rule_type=nengo.BCM(learning_rate=1e-9)):
        """Apply a plasticity rule to connections."""
        # This method needs to be adapted for BindsNET
        # Example: Toggle learning rules on/off or change parameters
        # self.plasticity_enabled = True/False
        # Access connection update rules and modify nu, etc.
        # if self.input_exc_conn: self.input_exc_conn.update_rule = ...
        logger.warning("apply_plasticity method needs adaptation for BindsNET")
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs shape: (batch_size, timesteps, features)
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]

        # We need to run the BindsNET sim for each item in the batch
        outputs = []
        for i in range(batch_size):
            # Add type ignore for TensorFlow Tensor -> NumPy array assignment
            input_signal: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = inputs[i].numpy() # type: ignore[assignment]
            # Ensure input_signal has correct shape/type for engine
            if input_signal.ndim == 1:
                # Assuming input layer expects (timesteps, features)
                # If input_signal is just (timesteps,), reshape or tile
                if self.input_layer:
                     input_features = self.input_layer.n
                     # Example: Repeat the signal across input features
                     input_signal = np.tile(input_signal[:, np.newaxis], (1, input_features))
                else:
                     logger.error("Input layer not found, cannot reshape signal.")
                     # Handle error appropriately, maybe return zeros
                     continue # Skip this batch item
            elif input_signal.ndim == 2:
                 # Check if features match input layer size
                 if self.input_layer and input_signal.shape[1] != self.input_layer.n:
                      logger.warning(f"Input signal features ({input_signal.shape[1]}) do not match input layer size ({self.input_layer.n}). Attempting to adapt.")
                      # Try padding or truncating (potential data loss)
                      target_features = self.input_layer.n
                      current_features = input_signal.shape[1]
                      if target_features > current_features:
                           padding = np.zeros((input_signal.shape[0], target_features - current_features))
                           input_signal = np.hstack((input_signal, padding))
                      else:
                           input_signal = input_signal[:, :target_features]
            else:
                 logger.error(f"Unsupported input signal dimensions: {input_signal.ndim}")
                 continue # Skip this batch item

            # Assuming temporal_window corresponds to timesteps
            duration = timesteps # BindsNET time is usually in steps
            self.network.reset_state_variables()
            # Run BindsNET simulation
            sim_inputs = {"input": torch.from_numpy(input_signal).float()}
            self.network.run(inputs=sim_inputs, time=duration)

            # Extract output probe data (adjust for BindsNET monitors)
            output_data_tensor = self.network.monitors["output"].get("s") # Get spike data
            if output_data_tensor is None:
                 # Handle case where monitor might be empty
                 output_data = np.zeros((duration, self.output_layer.n if self.output_layer else 0))
            else:
                 output_data = output_data_tensor.numpy()

            outputs.append(output_data)

        # Stack batch results and convert back to Tensor
        output_tensor = tf.stack([tf.convert_to_tensor(o, dtype=tf.float32) for o in outputs])
        # Ensure output has compatible shape e.g., (batch_size, timesteps, output_features)
        return output_tensor


def serve(port=50051):
    """Start the neuromorphic engine gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    neuromorphic_pb2_grpc.add_NeuromorphicEngineServicer_to_server(
        NeuromorphicEngineServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Neuromorphic engine server started on port {port}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Neuromorphic engine server stopped.")


if __name__ == "__main__":
    serve()
