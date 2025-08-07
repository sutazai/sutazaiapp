"""
Federated Learning System for SutazAI
====================================

Implements privacy-preserving distributed machine learning across AI agents.

Modules:
- coordinator: Central federated learning coordinator
- aggregator: Model aggregation algorithms (FedAvg, FedProx, FedOpt)
- client: Client-side training framework
- privacy: Differential privacy and secure aggregation
- versioning: Model versioning and rollback system
- monitoring: Performance monitoring and analytics
- dashboard: Federated learning dashboard
"""

from .coordinator import FederatedCoordinator
from .aggregator import FederatedAggregator
from .client import FederatedClient
from .privacy import PrivacyManager
from .versioning import ModelVersionManager
from .monitoring import FederatedMonitor
from .dashboard import FederatedDashboard

__all__ = [
    'FederatedCoordinator',
    'FederatedAggregator', 
    'FederatedClient',
    'PrivacyManager',
    'ModelVersionManager',
    'FederatedMonitor',
    'FederatedDashboard'
]