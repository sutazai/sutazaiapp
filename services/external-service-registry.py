#!/usr/bin/env python3
"""
Purpose: Central registry for all external services integrated with SutazAI
Usage: python external-service-registry.py [--list] [--add SERVICE] [--remove SERVICE]
Requirements: redis, consul-py
"""

import json
import redis
import consul
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

class ServiceRegistry:
    """Central registry for external service management"""
    
    def __init__(self, backend='consul'):
        self.backend = backend
        
        if backend == 'consul':
            self.client = consul.Consul(host='localhost', port=10040)
        elif backend == 'redis':
            self.client = redis.Redis(host='localhost', port=10110, decode_responses=True)
        else:
            # In-memory fallback
            self.client = {}
            
        self.service_prefix = 'sutazai/services/'
        
    def register_service(self, service_id: str, service_data: Dict[str, Any]) -> bool:
        """Register a new external service"""
        service_data['registered_at'] = datetime.utcnow().isoformat()
        service_data['status'] = 'active'
        
        try:
            if self.backend == 'consul':
                self.client.kv.put(
                    f"{self.service_prefix}{service_id}",
                    json.dumps(service_data)
                )
            elif self.backend == 'redis':
                self.client.hset(
                    'sutazai:services',
                    service_id,
                    json.dumps(service_data)
                )
            else:
                self.client[service_id] = service_data
                
            return True
            
        except Exception as e:
            print(f"Failed to register service: {e}")
            return False
    
    def get_service(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service details"""
        try:
            if self.backend == 'consul':
                _, data = self.client.kv.get(f"{self.service_prefix}{service_id}")
                if data:
                    return json.loads(data['Value'].decode())
                    
            elif self.backend == 'redis':
                data = self.client.hget('sutazai:services', service_id)
                if data:
                    return json.loads(data)
                    
            else:
                return self.client.get(service_id)
                
        except Exception as e:
            print(f"Failed to get service: {e}")
            
        return None
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services"""
        services = []
        
        try:
            if self.backend == 'consul':
                _, data = self.client.kv.get(self.service_prefix, recurse=True)
                if data:
                    for item in data:
                        service_data = json.loads(item['Value'].decode())
                        service_data['id'] = item['Key'].replace(self.service_prefix, '')
                        services.append(service_data)
                        
            elif self.backend == 'redis':
                for service_id, data in self.client.hgetall('sutazai:services').items():
                    service_data = json.loads(data)
                    service_data['id'] = service_id
                    services.append(service_data)
                    
            else:
                for service_id, data in self.client.items():
                    data['id'] = service_id
                    services.append(data)
                    
        except Exception as e:
            print(f"Failed to list services: {e}")
            
        return services
    
    def update_service_status(self, service_id: str, status: str) -> bool:
        """Update service status"""
        service = self.get_service(service_id)
        if service:
            service['status'] = status
            service['last_updated'] = datetime.utcnow().isoformat()
            return self.register_service(service_id, service)
        return False
    
    def remove_service(self, service_id: str) -> bool:
        """Remove a service from registry"""
        try:
            if self.backend == 'consul':
                self.client.kv.delete(f"{self.service_prefix}{service_id}")
            elif self.backend == 'redis':
                self.client.hdel('sutazai:services', service_id)
            else:
                self.client.pop(service_id, None)
                
            return True
            
        except Exception as e:
            print(f"Failed to remove service: {e}")
            return False
    
    def get_services_by_type(self, service_type: str) -> List[Dict[str, Any]]:
        """Get all services of a specific type"""
        all_services = self.list_services()
        return [s for s in all_services if s.get('type') == service_type]
    
    def get_service_health(self, service_id: str) -> Dict[str, Any]:
        """Get service health status"""
        # This would integrate with health check endpoints
        return {
            'service_id': service_id,
            'healthy': True,
            'last_check': datetime.utcnow().isoformat(),
            'details': {}
        }

# Predefined service templates
SERVICE_TEMPLATES = {
    'postgresql': {
        'type': 'database',
        'adapter_port': 10100,
        'original_port': 5432,
        'protocol': 'postgresql',
        'features': ['connection_pooling', 'query_monitoring', 'slow_query_detection'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'mysql': {
        'type': 'database',
        'adapter_port': 10101,
        'original_port': 3306,
        'protocol': 'mysql',
        'features': ['connection_pooling', 'query_monitoring', 'replication_status'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'mongodb': {
        'type': 'database',
        'adapter_port': 10102,
        'original_port': 27017,
        'protocol': 'mongodb',
        'features': ['connection_pooling', 'aggregation_monitoring', 'index_analysis'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'redis': {
        'type': 'cache',
        'adapter_port': 10110,
        'original_port': 6379,
        'protocol': 'redis',
        'features': ['command_monitoring', 'memory_analysis', 'pub_sub_support'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'rabbitmq': {
        'type': 'message_queue',
        'adapter_port': 10120,
        'original_port': 5672,
        'management_port': 15672,
        'protocol': 'amqp',
        'features': ['queue_monitoring', 'exchange_management', 'consumer_tracking'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'kafka': {
        'type': 'message_queue',
        'adapter_port': 10122,
        'original_port': 9092,
        'protocol': 'kafka',
        'features': ['topic_management', 'consumer_group_monitoring', 'lag_tracking'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'elasticsearch': {
        'type': 'search',
        'adapter_port': 10130,
        'original_port': 9200,
        'protocol': 'http',
        'features': ['index_management', 'query_optimization', 'cluster_monitoring'],
        'metrics_enabled': True,
        'health_check': '/health'
    },
    'prometheus': {
        'type': 'monitoring',
        'adapter_port': 10140,
        'original_port': 9090,
        'protocol': 'http',
        'features': ['metric_aggregation', 'alert_routing', 'federation'],
        'metrics_enabled': True,
        'health_check': '/-/healthy'
    },
    'grafana': {
        'type': 'monitoring',
        'adapter_port': 10141,
        'original_port': 3000,
        'protocol': 'http',
        'features': ['dashboard_embedding', 'user_management', 'alert_management'],
        'metrics_enabled': True,
        'health_check': '/api/health'
    }
}

def main():
    parser = argparse.ArgumentParser(description='SutazAI External Service Registry')
    parser.add_argument('--backend', choices=['consul', 'redis', 'memory'], 
                       default='memory', help='Registry backend')
    parser.add_argument('--list', action='store_true', help='List all services')
    parser.add_argument('--add', help='Add a service (use template name)')
    parser.add_argument('--remove', help='Remove a service')
    parser.add_argument('--status', help='Get service status')
    parser.add_argument('--type', help='Filter by service type')
    parser.add_argument('--health', help='Check service health')
    
    args = parser.parse_args()
    
    registry = ServiceRegistry(backend=args.backend)
    
    if args.list:
        services = registry.list_services()
        if args.type:
            services = [s for s in services if s.get('type') == args.type]
            
        print(f"Registered Services ({len(services)}):")
        for service in services:
            print(f"\n- {service.get('id', 'Unknown')}")
            print(f"  Type: {service.get('type', 'Unknown')}")
            print(f"  Status: {service.get('status', 'Unknown')}")
            print(f"  Adapter Port: {service.get('adapter_port', 'N/A')}")
            print(f"  Original Port: {service.get('original_port', 'N/A')}")
            
    elif args.add:
        if args.add in SERVICE_TEMPLATES:
            template = SERVICE_TEMPLATES[args.add].copy()
            template['name'] = args.add
            
            if registry.register_service(args.add, template):
                print(f"Service '{args.add}' registered successfully")
                print(f"Adapter will be available at port {template['adapter_port']}")
            else:
                print(f"Failed to register service '{args.add}'")
        else:
            print(f"Unknown service template: {args.add}")
            print(f"Available templates: {', '.join(SERVICE_TEMPLATES.keys())}")
            
    elif args.remove:
        if registry.remove_service(args.remove):
            print(f"Service '{args.remove}' removed successfully")
        else:
            print(f"Failed to remove service '{args.remove}'")
            
    elif args.status:
        service = registry.get_service(args.status)
        if service:
            print(f"Service: {args.status}")
            print(json.dumps(service, indent=2))
        else:
            print(f"Service '{args.status}' not found")
            
    elif args.health:
        health = registry.get_service_health(args.health)
        print(f"Health status for '{args.health}':")
        print(json.dumps(health, indent=2))
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()