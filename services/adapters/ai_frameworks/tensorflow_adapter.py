"""
TensorFlow adapter for deep learning operations
"""
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional, Union
from ..base_adapter import ServiceAdapter
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TensorFlowAdapter(ServiceAdapter):
    """Adapter for TensorFlow framework operations"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TensorFlow", config)
        self.model_path = config.get('model_path', '/models/tensorflow')
        self.gpu_memory_limit = config.get('gpu_memory_limit', 4096)  # MB
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize TensorFlow environment"""
        try:
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    if self.gpu_memory_limit:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=self.gpu_memory_limit
                            )]
                        )
                        
                logger.info(f"TensorFlow initialized with {len(gpus)} GPU(s)")
            else:
                logger.info("TensorFlow initialized with CPU only")
                
        except Exception as e:
            logger.error(f"Failed to initialize TensorFlow: {str(e)}")
            raise
            
    async def _custom_health_check(self) -> bool:
        """Check TensorFlow health"""
        try:
            # Simple operation to verify TensorFlow is working
            test_tensor = tf.random.normal([10, 10])
            result = tf.reduce_sum(test_tensor)
            return True
        except:
            return False
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get TensorFlow capabilities"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        return {
            'service': 'TensorFlow',
            'type': 'ai_framework',
            'version': tf.__version__,
            'features': [
                'deep_learning',
                'neural_networks',
                'eager_execution',
                'graph_optimization',
                'distributed_training',
                'tflite_conversion',
                'tensorboard_integration'
            ],
            'devices': {
                'gpus': len(gpus),
                'gpu_names': [gpu.name for gpu in gpus],
                'cpu_count': len(tf.config.experimental.list_physical_devices('CPU'))
            },
            'loaded_models': list(self.loaded_models.keys())
        }
        
    async def load_model(self,
                        model_name: str,
                        model_path: str,
                        model_format: str = 'saved_model') -> Dict[str, Any]:
        """Load a TensorFlow model"""
        try:
            def _load():
                if model_format == 'saved_model':
                    model = tf.saved_model.load(model_path)
                elif model_format == 'h5':
                    model = tf.keras.models.load_model(model_path)
                elif model_format == 'tflite':
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    return interpreter
                else:
                    raise ValueError(f"Unknown model format: {model_format}")
                    
                return model
                
            # Run in thread pool
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(self.executor, _load)
            
            self.loaded_models[model_name] = {
                'model': model,
                'format': model_format
            }
            
            return {
                'success': True,
                'model_name': model_name,
                'format': model_format
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def inference(self,
                       model_name: str,
                       input_data: Union[np.ndarray, tf.Tensor],
                       signature: str = 'serving_default') -> Dict[str, Any]:
        """Run inference on a loaded model"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            model_format = model_info['format']
            
            def _inference():
                # Convert input to tensor if needed
                if isinstance(input_data, np.ndarray):
                    input_tensor = tf.constant(input_data)
                else:
                    input_tensor = input_data
                    
                if model_format == 'tflite':
                    # TFLite inference
                    interpreter = model
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    
                    output = interpreter.get_tensor(output_details[0]['index'])
                elif model_format == 'saved_model':
                    # SavedModel inference
                    infer = model.signatures[signature]
                    output = infer(input_tensor)
                    # Convert output dict to array
                    output_key = list(output.keys())[0]
                    output = output[output_key].numpy()
                else:
                    # Keras model inference
                    output = model.predict(input_tensor)
                    
                return output
                
            # Run in thread pool
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(self.executor, _inference)
            
            return {
                'success': True,
                'output': output.tolist(),
                'shape': list(output.shape)
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def train_model(self,
                         model_name: str,
                         train_data: tf.data.Dataset,
                         epochs: int = 10,
                         batch_size: int = 32,
                         optimizer: str = 'adam',
                         loss: str = 'mse') -> Dict[str, Any]:
        """Train a Keras model"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model_info = self.loaded_models[model_name]
            if model_info['format'] != 'h5':
                return {
                    'success': False,
                    'error': "Only Keras models (h5 format) can be trained"
                }
                
            model = model_info['model']
            
            def _train():
                # Compile model
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy']
                )
                
                # Train
                history = model.fit(
                    train_data,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                return history.history
                
            # Run in thread pool
            loop = asyncio.get_event_loop()
            history = await loop.run_in_executor(self.executor, _train)
            
            return {
                'success': True,
                'history': history
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def optimize_model(self,
                           model_name: str,
                           optimization: str = 'tflite',
                           quantization: bool = False) -> Dict[str, Any]:
        """Optimize model for deployment"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            
            def _optimize():
                if optimization == 'tflite':
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    
                    if quantization:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        
                    tflite_model = converter.convert()
                    
                    # Save optimized model
                    optimized_path = f"{self.model_path}/{model_name}_optimized.tflite"
                    with open(optimized_path, 'wb') as f:
                        f.write(tflite_model)
                        
                    return optimized_path, len(tflite_model)
                    
                else:
                    raise ValueError(f"Unknown optimization: {optimization}")
                    
            # Run in thread pool
            loop = asyncio.get_event_loop()
            path, size = await loop.run_in_executor(self.executor, _optimize)
            
            return {
                'success': True,
                'optimized_model_path': path,
                'size_bytes': size
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def save_model(self, model_name: str, save_path: str) -> Dict[str, Any]:
        """Save a model"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            model_format = model_info['format']
            
            def _save():
                if model_format == 'h5':
                    model.save(save_path)
                elif model_format == 'saved_model':
                    tf.saved_model.save(model, save_path)
                else:
                    raise ValueError(f"Cannot save {model_format} format")
                    
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, _save)
            
            return {
                'success': True,
                'saved_to': save_path
            }
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()