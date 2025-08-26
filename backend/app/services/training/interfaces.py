"""
Training service interfaces
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

class TrainingStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingConfig:
    """Configuration for training job"""
    model_name: str
    dataset_path: Optional[str] = None
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "/tmp/model_output"
    use_fp16: bool = False
    seed: int = 42
    distributed: bool = False
    additional_args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingResult:
    """Result from training job"""
    job_id: str
    status: TrainingStatus
    model_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)

class Trainer(ABC):
    """Abstract base class for model trainers"""
    
    @abstractmethod
    async def train(self, config: TrainingConfig) -> TrainingResult:
        """
        Start a training job
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingResult with job information
        """
        pass
    
    @abstractmethod
    async def get_status(self, job_id: str) -> TrainingResult:
        """
        Get status of a training job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Current training result/status
        """
        pass
    
    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a running training job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this trainer is available
        
        Returns:
            True if trainer can be used
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the training service is healthy
        
        Returns:
            True if service is available
        """
        pass