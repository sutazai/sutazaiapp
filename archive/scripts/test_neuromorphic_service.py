import grpc
from core.protobufs import neuromorphic_pb2
from core.protobufs import neuromorphic_pb2_grpc
import numpy as np
import pytest


@pytest.mark.skip(reason="Skipping gRPC test, requires neuromorphic service to be running.")
def test_spike_processing():
    channel = grpc.insecure_channel("localhost:50051")
    stub = neuromorphic_pb2_grpc.NeuromorphicEngineStub(channel)

    # Generate test spikes using Poisson distribution
    spike_data = bytes(np.random.poisson(0.1, 128).astype(np.uint8))

    response = stub.ProcessSpikes(
        neuromorphic_pb2.SpikeInput(
            encoded_spikes=spike_data,
            temporal_window=100,
            neuron_profile="default",
            context_tags={"test": "true"},
        )
    )

    print(f"Received activations: {response.activations[:5]}")
    print(f"Energy usage: {response.energy_usage}")


if __name__ == "__main__":
    test_spike_processing()
