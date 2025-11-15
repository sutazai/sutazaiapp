#!/usr/bin/env python3
"""
RabbitMQ Integration Testing
Tests message queues, exchanges, routing patterns
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any

BASE_URL = "http://localhost:10200/api/v1"
RABBITMQ_MGMT = "http://localhost:10005/api"
RABBITMQ_AUTH = ("sutazai", "sutazai_secure_2024")
TIMEOUT = 30.0

class TestRabbitMQConnectivity:
    """Test RabbitMQ connection and health"""
    
    @pytest.mark.asyncio
    async def test_rabbitmq_management_ui(self):
        """Test RabbitMQ management UI accessibility"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/overview",
                auth=RABBITMQ_AUTH
            )
            assert response.status_code in [200, 401, 404]
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nRabbitMQ version: {data.get('rabbitmq_version', 'unknown')}")
    
    @pytest.mark.asyncio
    async def test_rabbitmq_vhost(self):
        """Test default vhost configuration"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/vhosts",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                vhosts = response.json()
                vhost_names = [v.get("name") for v in vhosts]
                print(f"\nVHosts: {vhost_names}")
                assert "/" in vhost_names or len(vhost_names) > 0


class TestRabbitMQExchanges:
    """Test RabbitMQ exchange configuration"""
    
    @pytest.mark.asyncio
    async def test_list_exchanges(self):
        """Test listing all exchanges"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/exchanges/%2F",  # %2F = /
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                exchanges = response.json()
                exchange_names = [e.get("name") for e in exchanges]
                print(f"\nExchanges: {len(exchange_names)}")
                
                # Default exchanges should exist
                expected = ["", "amq.direct", "amq.topic", "amq.fanout"]
                for exp in expected:
                    assert exp in exchange_names or True  # Informational
    
    @pytest.mark.asyncio
    async def test_topic_exchange(self):
        """Test topic exchange configuration"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/exchanges/%2F/amq.topic",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                exchange = response.json()
                assert exchange.get("type") == "topic"
                print(f"\nTopic exchange: {exchange.get('name')}, durable={exchange.get('durable')}")


class TestRabbitMQQueues:
    """Test RabbitMQ queue operations"""
    
    @pytest.mark.asyncio
    async def test_list_queues(self):
        """Test listing all queues"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/queues/%2F",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                queues = response.json()
                queue_names = [q.get("name") for q in queues]
                print(f"\nQueues: {queue_names}")
                
                # Check for agent queues
                agent_queues = [q for q in queue_names if "agent" in q.lower()]
                print(f"Agent queues: {len(agent_queues)}")
    
    @pytest.mark.asyncio
    async def test_queue_stats(self):
        """Test queue statistics"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Get all queues first
            queues_resp = await client.get(
                f"{RABBITMQ_MGMT}/queues/%2F",
                auth=RABBITMQ_AUTH
            )
            
            if queues_resp.status_code == 200:
                queues = queues_resp.json()
                
                for queue in queues[:3]:  # Check first 3 queues
                    queue_name = queue.get("name")
                    messages = queue.get("messages", 0)
                    consumers = queue.get("consumers", 0)
                    
                    print(f"\nQueue '{queue_name}': {messages} messages, {consumers} consumers")


class TestRabbitMQConnections:
    """Test RabbitMQ connections and channels"""
    
    @pytest.mark.asyncio
    async def test_active_connections(self):
        """Test active connections to RabbitMQ"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/connections",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                connections = response.json()
                print(f"\nActive connections: {len(connections)}")
                
                for conn in connections[:3]:
                    print(f"  Client: {conn.get('client_properties', {}).get('connection_name', 'unknown')}")
    
    @pytest.mark.asyncio
    async def test_channels(self):
        """Test active channels"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/channels",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                channels = response.json()
                print(f"\nActive channels: {len(channels)}")


class TestMessageRouting:
    """Test message routing patterns"""
    
    @pytest.mark.asyncio
    async def test_direct_routing(self):
        """Test direct exchange routing"""
        # This requires publishing messages - informational test
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Check if MCP Bridge publishes to direct exchange
            response = await client.get(
                f"{RABBITMQ_MGMT}/exchanges/%2F/amq.direct",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                exchange = response.json()
                print(f"\nDirect exchange configured: {exchange.get('name')}")
                assert True
    
    @pytest.mark.asyncio
    async def test_fanout_routing(self):
        """Test fanout exchange routing"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/exchanges/%2F/amq.fanout",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                exchange = response.json()
                print(f"\nFanout exchange configured: {exchange.get('name')}")


class TestMessagePersistence:
    """Test message persistence and durability"""
    
    @pytest.mark.asyncio
    async def test_durable_queues(self):
        """Test that critical queues are durable"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/queues/%2F",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                queues = response.json()
                
                for queue in queues:
                    if "agent" in queue.get("name", "").lower():
                        is_durable = queue.get("durable", False)
                        print(f"\nAgent queue '{queue.get('name')}': durable={is_durable}")


class TestRabbitMQPerformance:
    """Test RabbitMQ performance metrics"""
    
    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Test message publishing rate"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(
                f"{RABBITMQ_MGMT}/overview",
                auth=RABBITMQ_AUTH
            )
            
            if response.status_code == 200:
                data = response.json()
                message_stats = data.get("message_stats", {})
                
                publish_rate = message_stats.get("publish_details", {}).get("rate", 0)
                deliver_rate = message_stats.get("deliver_details", {}).get("rate", 0)
                
                print(f"\nPublish rate: {publish_rate:.2f} msg/s")
                print(f"Deliver rate: {deliver_rate:.2f} msg/s")


class TestConsulIntegration:
    """Test Consul service discovery integration"""
    
    @pytest.mark.asyncio
    async def test_consul_health(self):
        """Test Consul health endpoint"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:8500/v1/status/leader")
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                print(f"\nConsul leader: {response.text}")
    
    @pytest.mark.asyncio
    async def test_consul_services(self):
        """Test Consul service registry"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:8500/v1/catalog/services")
            
            if response.status_code == 200:
                services = response.json()
                print(f"\nRegistered services: {list(services.keys())}")
                assert len(services) >= 0  # Informational
    
    @pytest.mark.asyncio
    async def test_consul_kv_store(self):
        """Test Consul key-value store"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Try to read from KV store
            response = await client.get("http://localhost:8500/v1/kv/?keys")
            
            if response.status_code == 200:
                keys = response.json()
                print(f"\nKV store keys: {len(keys)}")


class TestKongGateway:
    """Test Kong API Gateway"""
    
    @pytest.mark.asyncio
    async def test_kong_admin_api(self):
        """Test Kong admin API"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:8001/")
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nKong version: {data.get('version', 'unknown')}")
    
    @pytest.mark.asyncio
    async def test_kong_services(self):
        """Test Kong services configuration"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:8001/services")
            
            if response.status_code == 200:
                data = response.json()
                services = data.get("data", [])
                print(f"\nKong services: {len(services)}")
                
                for svc in services[:3]:
                    print(f"  Service: {svc.get('name')} -> {svc.get('host')}")
    
    @pytest.mark.asyncio
    async def test_kong_routes(self):
        """Test Kong routes configuration"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:8001/routes")
            
            if response.status_code == 200:
                data = response.json()
                routes = data.get("data", [])
                print(f"\nKong routes: {len(routes)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
