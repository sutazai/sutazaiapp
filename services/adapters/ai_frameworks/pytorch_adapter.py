"""
PyTorch adapter for deep learning model operations
"""
import torch
import torch.nn as nn
import numpy as np
from ..base_adapter import ServiceAdapter
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PyTorchAdapter(ServiceAdapter):
    """Adapter for PyTorch framework operations"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PyTorch", config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = config.get('model_path', '/models/pytorch')
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize PyTorch environment"""
        try:
            # Set device
            self.device = torch.device(self.device)
            
            # Log GPU information if available
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")
                    
            logger.info(f"PyTorch initialized with device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch: {str(e)}")
            raise
            
    async def _custom_health_check(self) -> bool:
        """Check PyTorch health"""
        try:
            # Simple tensor operation to verify PyTorch is working
            test_tensor = torch.randn(10, 10).to(self.device)
            result = torch.sum(test_tensor)
            return True
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return False
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get PyTorch capabilities"""
        return {
            'service': 'PyTorch',
            'type': 'ai_framework',
            'version': torch.__version__,
            'features': [
                'deep_learning',
                'neural_networks',
                'automatic_differentiation',
                'gpu_acceleration',
                'distributed_training',
                'quantization',
                'jit_compilation'
            ],
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'loaded_models': list(self.loaded_models.keys())
        }
        
    async def load_model(self, 
                        model_name: str,
                        model_path: str,
                        model_class: Optional[type] = None) -> Dict[str, Any]:
        """Load a PyTorch model"""
        try:
            def _load():
                if model_class:
                    # Load model with architecture
                    model = model_class()
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                else:
                    # Load entire model
                    model = torch.load(model_path, map_location=self.device)
                    
                model.to(self.device)
                model.eval()
                return model
                
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(self.executor, _load)
            
            self.loaded_models[model_name] = model
            
            return {
                'success': True,
                'model_name': model_name,
                'device': str(self.device),
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def inference(self,
                       model_name: str,
                       input_data: Union[np.ndarray, torch.Tensor],
                       batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Run inference on a loaded model"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model = self.loaded_models[model_name]
            
            def _inference():
                # Convert input to tensor if needed
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).to(self.device)
                else:
                    input_tensor = input_data.to(self.device)
                    
                # Run inference
                with torch.no_grad():
                    if batch_size:
                        # Batch inference
                        outputs = []
                        for i in range(0, len(input_tensor), batch_size):
                            batch = input_tensor[i:i + batch_size]
                            output = model(batch)
                            outputs.append(output.cpu())
                        output = torch.cat(outputs, dim=0)
                    else:
                        output = model(input_tensor).cpu()
                        
                return output.numpy()
                
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
            
    async def train_step(self,
                        model_name: str,
                        inputs: torch.Tensor,
                        targets: torch.Tensor,
                        loss_fn: str = 'mse',
                        optimizer: str = 'adam',
                        lr: float = 0.001) -> Dict[str, Any]:
        """Perform a single training step"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model = self.loaded_models[model_name]
            
            def _train_step():
                # Set model to training mode
                model.train()
                
                # Create optimizer
                if optimizer == 'adam':
                    opt = torch.optim.Adam(model.parameters(), lr=lr)
                elif optimizer == 'sgd':
                    opt = torch.optim.SGD(model.parameters(), lr=lr)
                else:
                    raise ValueError(f"Unknown optimizer: {optimizer}")
                    
                # Create loss function
                if loss_fn == 'mse':
                    criterion = nn.MSELoss()
                elif loss_fn == 'cross_entropy':
                    criterion = nn.CrossEntropyLoss()
                else:
                    raise ValueError(f"Unknown loss function: {loss_fn}")
                    
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                return loss.item()
                
            # Run in thread pool
            loop = asyncio.get_event_loop()
            loss = await loop.run_in_executor(self.executor, _train_step)
            
            return {
                'success': True,
                'loss': loss
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def save_model(self, model_name: str, save_path: str) -> Dict[str, Any]:
        """Save a loaded model"""
        try:
            if model_name not in self.loaded_models:
                return {
                    'success': False,
                    'error': f"Model {model_name} not loaded"
                }
                
            model = self.loaded_models[model_name]
            
            def _save():
                torch.save(model.state_dict(), save_path)
                
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
            
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model"""
        if model_name not in self.loaded_models:
            return {
                'error': f"Model {model_name} not loaded"
            }
            
        model = self.loaded_models[model_name]
        
        return {
            'model_name': model_name,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': str(next(model.parameters()).device),
            'layers': len(list(model.modules()))
        }
        
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()