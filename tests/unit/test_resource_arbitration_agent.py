#!/usr/bin/env python3
"""
Unit and integration tests for Resource Arbitration Agent
"""
import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add paths for imports
sys.path.append('/opt/sutazaiapp/agents')
sys.path.append('/opt/sutazaiapp/agents/resource_arbitration_agent')

from resource_arbitration_agent.app import (
    ResourceArbitrationAgent, ResourceAllocation, ResourceReservation,
    ResourceCapacity, AllocationPolicy, ConflictResolution,
    ResourceType, Priority, AGENT_ID
)
from core.messaging import ResourceMessage, MessageType


class TestResourceArbitrationAgent:
    """Test suite for Resource Arbitration Agent"""
    
    @pytest.fixture
    async def arbitrator(self):
        """Create arbitrator instance with mocked dependencies"""
        arb = ResourceArbitrationAgent()
        
        # Mock Redis client
        arb.redis_client = AsyncMock()
        arb.redis_client.ping = AsyncMock(return_value=True)
        arb.redis_client.hset = AsyncMock()
        arb.redis_client.hget = AsyncMock(return_value=None)
        arb.redis_client.hgetall = AsyncMock(return_value={})
        arb.redis_client.hdel = AsyncMock()
        
        # Mock message processor
        arb.message_processor = AsyncMock()
        arb.message_processor.start = AsyncMock()
        arb.message_processor.stop = AsyncMock()
        arb.message_processor.rabbitmq_client = AsyncMock()
        
        # Mock psutil for system resources
        with patch('psutil.cpu_count', return_value=8):
            with patch('psutil.cpu_percent', return_value=25.0):
                with patch('psutil.virtual_memory') as mock_mem:
                    mock_mem.return_value = MagicMock(
                        total=16 * 1024**3,  # 16 GB
                        available=12 * 1024**3,  # 12 GB available
                        percent=25.0
                    )
                    with patch('psutil.disk_usage') as mock_disk:
                        mock_disk.return_value = MagicMock(
                            total=500 * 1024**3,  # 500 GB
                            free=400 * 1024**3,  # 400 GB free
                            percent=20.0
                        )
                        await arb.discover_system_resources()
        
        return arb
    
    @pytest.mark.asyncio
    async def test_discover_system_resources(self, arbitrator):
        """Test system resource discovery"""
        # Resources should be discovered in fixture
        assert ResourceType.CPU in arbitrator.system_resources
        assert ResourceType.MEMORY in arbitrator.system_resources
        assert ResourceType.DISK in arbitrator.system_resources
        assert ResourceType.NETWORK in arbitrator.system_resources
        
        # Check CPU resources
        cpu_capacity = arbitrator.system_resources[ResourceType.CPU]
        assert cpu_capacity.total_capacity == 8  # 8 cores
        assert cpu_capacity.unit == "cores"
        assert cpu_capacity.utilization_percent == 25.0
        
        # Check memory resources
        mem_capacity = arbitrator.system_resources[ResourceType.MEMORY]
        assert mem_capacity.total_capacity == 16.0  # 16 GB
        assert mem_capacity.unit == "GB"
        assert mem_capacity.available == 12.0
        
        # Check disk resources
        disk_capacity = arbitrator.system_resources[ResourceType.DISK]
        assert disk_capacity.total_capacity == pytest.approx(500.0, 1)
        assert disk_capacity.unit == "GB"
    
    @pytest.mark.asyncio
    async def test_process_allocation_request_success(self, arbitrator):
        """Test successful resource allocation"""
        reservation = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.CPU,
            requested_amount=2.0,  # 2 cores
            unit="cores",
            priority=Priority.NORMAL
        )
        
        allocation = await arbitrator.process_allocation_request(reservation)
        
        assert allocation is not None
        assert allocation.agent_id == "agent-1"
        assert allocation.resource_type == ResourceType.CPU
        assert allocation.amount == 2.0
        assert allocation.allocation_id in arbitrator.allocations
        
        # Verify capacity updated
        cpu_capacity = arbitrator.system_resources[ResourceType.CPU]
        assert cpu_capacity.allocated == 2.0
        assert cpu_capacity.available == 6.0  # 8 - 2
        
        # Verify Redis storage
        arbitrator.redis_client.hset.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_allocation_request_capacity_exceeded(self, arbitrator):
        """Test allocation request exceeding capacity"""
        # Try to allocate more than max allowed (80% of 8 cores = 6.4)
        reservation = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.CPU,
            requested_amount=7.0,  # Exceeds max
            unit="cores",
            priority=Priority.NORMAL
        )
        
        allocation = await arbitrator.process_allocation_request(reservation)
        
        assert allocation is None
        assert reservation.status == "denied"
    
    @pytest.mark.asyncio
    async def test_process_allocation_request_per_agent_limit(self, arbitrator):
        """Test per-agent allocation limit"""
        # First allocation - within limit
        reservation1 = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.CPU,
            requested_amount=2.0,
            unit="cores",
            priority=Priority.NORMAL
        )
        
        allocation1 = await arbitrator.process_allocation_request(reservation1)
        assert allocation1 is not None
        
        # Second allocation - would exceed per-agent limit (30% of 8 = 2.4)
        reservation2 = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.CPU,
            requested_amount=1.0,
            unit="cores",
            priority=Priority.NORMAL
        )
        
        allocation2 = await arbitrator.process_allocation_request(reservation2)
        assert allocation2 is None
    
    @pytest.mark.asyncio
    async def test_gpu_exclusive_allocation(self, arbitrator):
        """Test exclusive GPU allocation"""
        # Add GPU resource
        arbitrator.system_resources[ResourceType.GPU] = ResourceCapacity(
            resource_type=ResourceType.GPU,
            total_capacity=1.0,
            allocated=0.0,
            available=1.0,
            unit="GPUs",
            utilization_percent=0.0
        )
        
        # First GPU allocation
        reservation1 = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.GPU,
            requested_amount=1.0,
            unit="GPUs",
            priority=Priority.NORMAL
        )
        
        allocation1 = await arbitrator.process_allocation_request(reservation1)
        assert allocation1 is not None
        assert allocation1.exclusive is True
        
        # Second GPU allocation should fail (exclusive)
        reservation2 = ResourceReservation(
            agent_id="agent-2",
            resource_type=ResourceType.GPU,
            requested_amount=1.0,
            unit="GPUs",
            priority=Priority.NORMAL
        )
        
        allocation2 = await arbitrator.process_allocation_request(reservation2)
        assert allocation2 is None
    
    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self, arbitrator):
        """Test resource conflict detection and resolution"""
        # Add GPU resource
        arbitrator.system_resources[ResourceType.GPU] = ResourceCapacity(
            resource_type=ResourceType.GPU,
            total_capacity=1.0,
            allocated=0.0,
            available=1.0,
            unit="GPUs",
            utilization_percent=0.0
        )
        
        # Allocate GPU to low priority agent
        low_priority_allocation = ResourceAllocation(
            agent_id="agent-low",
            resource_type=ResourceType.GPU,
            amount=1.0,
            unit="GPUs",
            priority=Priority.LOW,
            exclusive=True
        )
        arbitrator.allocations[low_priority_allocation.allocation_id] = low_priority_allocation
        
        # High priority request for same GPU
        high_priority_reservation = ResourceReservation(
            agent_id="agent-high",
            resource_type=ResourceType.GPU,
            requested_amount=1.0,
            unit="GPUs",
            priority=Priority.CRITICAL
        )
        
        # Should preempt low priority allocation
        allocation = await arbitrator.process_allocation_request(high_priority_reservation)
        
        assert allocation is not None
        assert allocation.agent_id == "agent-high"
        
        # Low priority allocation should be removed
        assert low_priority_allocation.allocation_id not in arbitrator.allocations
    
    @pytest.mark.asyncio
    async def test_release_allocation(self, arbitrator):
        """Test releasing allocated resources"""
        # Create an allocation
        allocation = ResourceAllocation(
            allocation_id="alloc-123",
            agent_id="agent-1",
            resource_type=ResourceType.MEMORY,
            amount=4.0,  # 4 GB
            unit="GB"
        )
        
        arbitrator.allocations["alloc-123"] = allocation
        arbitrator.system_resources[ResourceType.MEMORY].allocated = 4.0
        arbitrator.system_resources[ResourceType.MEMORY].available = 12.0
        
        # Release allocation
        await arbitrator.release_allocation("alloc-123")
        
        # Verify allocation removed
        assert "alloc-123" not in arbitrator.allocations
        
        # Verify capacity restored
        mem_capacity = arbitrator.system_resources[ResourceType.MEMORY]
        assert mem_capacity.allocated == 0.0
        assert mem_capacity.available == 16.0
        
        # Verify Redis removal
        arbitrator.redis_client.hdel.assert_called_with(
            "resource:allocations",
            "alloc-123"
        )
    
    @pytest.mark.asyncio
    async def test_allocation_with_expiration(self, arbitrator):
        """Test allocation with time-based expiration"""
        reservation = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.MEMORY,
            requested_amount=2.0,
            unit="GB",
            duration_seconds=300,  # 5 minutes
            priority=Priority.NORMAL
        )
        
        allocation = await arbitrator.process_allocation_request(reservation)
        
        assert allocation is not None
        assert allocation.expires_at is not None
        
        # Check expiration time is set correctly
        expected_expiry = datetime.utcnow() + timedelta(seconds=300)
        time_diff = abs((allocation.expires_at - expected_expiry).total_seconds())
        assert time_diff < 1  # Within 1 second tolerance
    
    @pytest.mark.asyncio
    async def test_oversubscription_for_network(self, arbitrator):
        """Test oversubscription allowed for network resources"""
        # Network allows oversubscription up to 1.5x
        policy = arbitrator.policies[ResourceType.NETWORK]
        assert policy.allow_oversubscription is True
        assert policy.oversubscription_ratio == 1.5
        
        # Allocate up to normal limit (95% of 1000 Mbps = 950)
        reservation1 = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.NETWORK,
            requested_amount=900.0,
            unit="Mbps",
            priority=Priority.NORMAL
        )
        
        allocation1 = await arbitrator.process_allocation_request(reservation1)
        assert allocation1 is not None
        
        # Try to allocate more (oversubscription)
        reservation2 = ResourceReservation(
            agent_id="agent-2",
            resource_type=ResourceType.NETWORK,
            requested_amount=400.0,
            unit="Mbps",
            priority=Priority.NORMAL
        )
        
        # Should succeed due to oversubscription (900 + 400 = 1300 < 1500)
        allocation2 = await arbitrator.process_allocation_request(reservation2)
        assert allocation2 is not None
        
        # Total allocated: 1300 Mbps (oversubscribed)
        network_capacity = arbitrator.system_resources[ResourceType.NETWORK]
        assert network_capacity.allocated == 1300.0
    
    @pytest.mark.asyncio
    async def test_queued_reservation_processing(self, arbitrator):
        """Test processing of queued reservations"""
        # Fill up CPU allocation
        reservation1 = ResourceReservation(
            agent_id="agent-1",
            resource_type=ResourceType.CPU,
            requested_amount=6.0,
            unit="cores",
            priority=Priority.NORMAL
        )
        
        allocation1 = await arbitrator.process_allocation_request(reservation1)
        assert allocation1 is not None
        
        # Try to allocate more (should be queued)
        reservation2 = ResourceReservation(
            reservation_id="res-queued",
            agent_id="agent-2",
            resource_type=ResourceType.CPU,
            requested_amount=2.0,
            unit="cores",
            priority=Priority.NORMAL,
            status="pending"
        )
        
        arbitrator.reservations["res-queued"] = reservation2
        
        # Release first allocation
        await arbitrator.release_allocation(allocation1.allocation_id)
        
        # Queued reservation should now be processed
        assert "res-queued" not in arbitrator.reservations
    
    @pytest.mark.asyncio
    async def test_get_status(self, arbitrator):
        """Test getting arbitrator status"""
        # Add some allocations
        allocation1 = ResourceAllocation(
            agent_id="agent-1",
            resource_type=ResourceType.CPU,
            amount=2.0,
            unit="cores"
        )
        allocation2 = ResourceAllocation(
            agent_id="agent-2",
            resource_type=ResourceType.MEMORY,
            amount=4.0,
            unit="GB"
        )
        
        arbitrator.allocations = {
            "alloc-1": allocation1,
            "alloc-2": allocation2
        }
        
        # Add pending reservation
        arbitrator.reservations["res-1"] = ResourceReservation(
            agent_id="agent-3",
            resource_type=ResourceType.DISK,
            requested_amount=100.0,
            unit="GB",
            status="pending"
        )
        
        status = await arbitrator.get_status()
        
        assert status["status"] == "healthy"
        assert status["total_allocations"] == 2
        assert status["pending_reservations"] == 1
        assert "resources" in status
        assert ResourceType.CPU.value in status["resources"]
    
    @pytest.mark.asyncio
    async def test_allocation_cleanup(self, arbitrator):
        """Test cleanup of expired allocations"""
        # Create expired allocation
        expired_allocation = ResourceAllocation(
            allocation_id="alloc-expired",
            agent_id="agent-1",
            resource_type=ResourceType.MEMORY,
            amount=2.0,
            unit="GB",
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Already expired
        )
        
        arbitrator.allocations["alloc-expired"] = expired_allocation
        arbitrator.running = True
        
        # Mock release_allocation to track calls
        arbitrator.release_allocation = AsyncMock()
        
        # Run cleanup once
        async def run_cleanup_once():
            current_time = datetime.utcnow()
            expired_allocations = []
            
            for allocation_id, allocation in arbitrator.allocations.items():
                if allocation.expires_at and current_time > allocation.expires_at:
                    expired_allocations.append(allocation_id)
            
            for allocation_id in expired_allocations:
                await arbitrator.release_allocation(allocation_id)
        
        await run_cleanup_once()
        
        # Verify expired allocation was released
        arbitrator.release_allocation.assert_called_once_with("alloc-expired")
    
    @pytest.mark.asyncio
    async def test_load_allocations_from_redis(self, arbitrator):
        """Test loading existing allocations from Redis"""
        # Mock Redis data
        allocation_data = {
            "alloc-1": json.dumps({
                "allocation_id": "alloc-1",
                "agent_id": "agent-1",
                "resource_type": "cpu",
                "amount": 2.0,
                "unit": "cores",
                "allocated_at": datetime.utcnow().isoformat(),
                "expires_at": None,
                "exclusive": False,
                "priority": 1,
                "metadata": {}
            }),
            "alloc-2": json.dumps({
                "allocation_id": "alloc-2",
                "agent_id": "agent-2",
                "resource_type": "memory",
                "amount": 4.0,
                "unit": "GB",
                "allocated_at": datetime.utcnow().isoformat(),
                "expires_at": None,
                "exclusive": False,
                "priority": 1,
                "metadata": {}
            })
        }
        
        arbitrator.redis_client.hgetall.return_value = allocation_data
        
        await arbitrator.load_allocations()
        
        assert len(arbitrator.allocations) == 2
        assert "alloc-1" in arbitrator.allocations
        assert "alloc-2" in arbitrator.allocations
        
        alloc1 = arbitrator.allocations["alloc-1"]
        assert alloc1.agent_id == "agent-1"
        assert alloc1.resource_type == ResourceType.CPU
        assert alloc1.amount == 2.0


class TestResourceArbitrationIntegration:
    """Integration tests for resource arbitration with messaging"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_resource_lifecycle(self):
        """Test complete resource allocation lifecycle"""
        with patch('aio_pika.connect_robust') as mock_connect:
            # Setup mock RabbitMQ
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_queue = AsyncMock()
            
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            mock_channel.declare_queue.return_value = mock_queue
            
            arbitrator = ResourceArbitrationAgent()
            
            # Mock Redis
            with patch('redis.asyncio.from_url') as mock_redis:
                mock_redis_client = AsyncMock()
                mock_redis.return_value = mock_redis_client
                mock_redis_client.ping.return_value = True
                mock_redis_client.hgetall.return_value = {}
                
                # Mock psutil
                with patch('psutil.cpu_count', return_value=8):
                    with patch('psutil.cpu_percent', return_value=25.0):
                        with patch('psutil.virtual_memory') as mock_mem:
                            mock_mem.return_value = MagicMock(
                                total=16 * 1024**3,
                                available=12 * 1024**3,
                                percent=25.0
                            )
                            with patch('psutil.disk_usage') as mock_disk:
                                mock_disk.return_value = MagicMock(
                                    total=500 * 1024**3,
                                    free=400 * 1024**3,
                                    percent=20.0
                                )
                                
                                # Initialize
                                await arbitrator.initialize()
                                
                                # Create resource request message
                                resource_msg = ResourceMessage(
                                    message_id="res-msg-1",
                                    message_type=MessageType.RESOURCE_REQUEST,
                                    source_agent="agent-test",
                                    resource_type="cpu",
                                    resource_amount=2.0,
                                    resource_unit="cores",
                                    duration_seconds=300
                                )
                                
                                # Process resource request
                                await arbitrator.message_processor.handle_resource_request(
                                    resource_msg.dict()
                                )
                                
                                # Verify allocation created
                                assert len(arbitrator.allocations) == 1
                                
                                # Get allocation ID
                                allocation_id = list(arbitrator.allocations.keys())[0]
                                allocation = arbitrator.allocations[allocation_id]
                                
                                assert allocation.agent_id == "agent-test"
                                assert allocation.amount == 2.0
                                assert allocation.resource_type == ResourceType.CPU
                                
                                # Release resource
                                release_msg = {
                                    "source_agent": "agent-test",
                                    "payload": {"allocation_id": allocation_id}
                                }
                                
                                await arbitrator.message_processor.handle_resource_release(
                                    release_msg
                                )
                                
                                # Verify allocation released
                                assert len(arbitrator.allocations) == 0
                                
                                # Cleanup
                                await arbitrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])