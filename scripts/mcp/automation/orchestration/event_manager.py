#!/usr/bin/env python3
"""
MCP Event Management System

Event-driven architecture implementation for MCP orchestration. Provides
publish-subscribe patterns, event routing, filtering, and asynchronous
event processing with priority queues and delivery guarantees.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 11:56:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import defaultdict, deque
import logging
from asyncio import PriorityQueue

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MCPAutomationConfig


class EventType(Enum):
    """Event types in the MCP system."""
    SYSTEM = "system"              # System-level events
    WORKFLOW = "workflow"          # Workflow execution events
    SERVICE = "service"            # Service status events
    UPDATE = "update"              # Update operation events
    CLEANUP = "cleanup"            # Cleanup operation events
    ALERT = "alert"                # Alert and warning events
    NOTIFICATION = "notification"  # User notifications
    METRIC = "metric"              # Performance metrics
    AUDIT = "audit"                # Audit trail events
    ERROR = "error"                # Error events
    CUSTOM = "custom"              # Custom application events


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 4
    NORMAL = 3
    HIGH = 2
    CRITICAL = 1
    EMERGENCY = 0
    
    def __lt__(self, other):
        """Enable priority comparison."""
        if isinstance(other, EventPriority):
            return self.value < other.value
        return NotImplemented


class DeliveryMode(Enum):
    """Event delivery modes."""
    FIRE_AND_FORGET = "fire_and_forget"  # No delivery guarantee
    AT_LEAST_ONCE = "at_least_once"      # Guaranteed delivery with possible duplicates
    AT_MOST_ONCE = "at_most_once"        # Single delivery attempt
    EXACTLY_ONCE = "exactly_once"        # Guaranteed single delivery (requires ack)


@dataclass
class Event:
    """Event data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    def __lt__(self, other):
        """Enable priority queue sorting."""
        if isinstance(other, Event):
            return self.priority < other.priority
        return NotImplemented
        
    def is_expired(self) -> bool:
        """Check if event has expired."""
        if self.ttl:
            age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
            return age > self.ttl
        return False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl,
            "delivery_mode": self.delivery_mode.value
        }


@dataclass
class EventFilter:
    """Event filter criteria."""
    types: Optional[Set[EventType]] = None
    sources: Optional[Set[str]] = None
    priorities: Optional[Set[EventPriority]] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria."""
        if self.types and event.type not in self.types:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.priorities and event.priority not in self.priorities:
            return False
            
        # Check metadata filters
        for key, value in self.metadata_filters.items():
            if key not in event.metadata or event.metadata[key] != value:
                return False
                
        return True


@dataclass
class Subscription:
    """Event subscription information."""
    id: str
    subscriber: str
    handler: Callable
    filter: EventFilter
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivery_count: int = 0
    error_count: int = 0
    last_delivery: Optional[datetime] = None
    last_error: Optional[str] = None


class EventBus:
    """
    Central event bus for publish-subscribe communication.
    
    Manages event routing, filtering, and delivery with support for
    priority queues, async handlers, and delivery guarantees.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize event bus."""
        self.config = config or MCPAutomationConfig()
        self.logger = self._setup_logging()
        
        # Event storage
        self._event_queue: PriorityQueue = PriorityQueue()
        self._event_history: deque = deque(maxlen=1000)
        self._pending_events: Dict[str, Event] = {}
        
        # Subscriptions
        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._subscription_index: Dict[str, Subscription] = {}
        
        # Processing
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._paused = False
        
        # Metrics
        self.metrics = {
            "events_published": 0,
            "events_delivered": 0,
            "events_failed": 0,
            "events_expired": 0,
            "active_subscriptions": 0
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("mcp.event_bus")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def initialize(self) -> None:
        """Initialize event bus."""
        self.logger.info("Initializing event bus...")
        
        # Start processing task
        self._processing_task = asyncio.create_task(self._process_events())
        
        self.logger.info("Event bus initialized")
        
    async def publish(self, event: Event) -> str:
        """Publish an event to the bus."""
        try:
            # Validate event
            if event.is_expired():
                self.logger.warning(f"Attempted to publish expired event: {event.id}")
                self.metrics["events_expired"] += 1
                return event.id
                
            # Add to queue
            await self._event_queue.put((event.priority.value, event))
            self.metrics["events_published"] += 1
            
            # Store for delivery tracking
            if event.delivery_mode in [DeliveryMode.AT_LEAST_ONCE, DeliveryMode.EXACTLY_ONCE]:
                self._pending_events[event.id] = event
                
            self.logger.debug(f"Published event: {event.id} ({event.type.value})")
            
            return event.id
            
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            raise
            
    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable,
        filter: Optional[EventFilter] = None,
        subscriber_name: Optional[str] = None
    ) -> str:
        """Subscribe to events."""
        try:
            subscription = Subscription(
                id=str(uuid.uuid4()),
                subscriber=subscriber_name or handler.__name__,
                handler=handler,
                filter=filter or EventFilter()
            )
            
            self._subscriptions[event_type].append(subscription)
            self._subscription_index[subscription.id] = subscription
            self.metrics["active_subscriptions"] += 1
            
            self.logger.info(
                f"Registered subscription: {subscription.subscriber} -> {event_type.value}"
            )
            
            return subscription.id
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe: {e}")
            raise
            
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        if subscription_id in self._subscription_index:
            subscription = self._subscription_index[subscription_id]
            
            # Remove from type subscriptions
            for event_type, subs in self._subscriptions.items():
                if subscription in subs:
                    subs.remove(subscription)
                    
            # Remove from index
            del self._subscription_index[subscription_id]
            self.metrics["active_subscriptions"] -= 1
            
            self.logger.info(f"Unsubscribed: {subscription.subscriber}")
            return True
            
        return False
        
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while not self._shutdown_event.is_set():
            try:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get next event with timeout
                try:
                    priority, event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # Check expiration
                if event.is_expired():
                    self.logger.debug(f"Event expired: {event.id}")
                    self.metrics["events_expired"] += 1
                    continue
                    
                # Process event
                await self._deliver_event(event)
                
                # Add to history
                self._event_history.append(event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                
    async def _deliver_event(self, event: Event) -> None:
        """Deliver event to subscribers."""
        try:
            # Get matching subscriptions
            subscriptions = self._subscriptions.get(event.type, [])
            
            # Also check CUSTOM type subscriptions for all events
            if event.type != EventType.CUSTOM:
                subscriptions.extend(self._subscriptions.get(EventType.CUSTOM, []))
                
            delivered = False
            tasks = []
            
            for subscription in subscriptions:
                if not subscription.active:
                    continue
                    
                # Apply filter
                if not subscription.filter.matches(event):
                    continue
                    
                # Create delivery task
                task = asyncio.create_task(
                    self._deliver_to_subscriber(event, subscription)
                )
                tasks.append(task)
                delivered = True
                
            # Wait for all deliveries
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                failures = [r for r in results if isinstance(r, Exception)]
                if failures:
                    self.logger.warning(f"Event delivery had {len(failures)} failures")
                    
            if delivered:
                self.metrics["events_delivered"] += 1
                
                # Remove from pending if exactly-once delivery
                if event.delivery_mode == DeliveryMode.EXACTLY_ONCE:
                    self._pending_events.pop(event.id, None)
            else:
                self.logger.debug(f"No subscribers for event: {event.id}")
                
        except Exception as e:
            self.logger.error(f"Event delivery error: {e}")
            self.metrics["events_failed"] += 1
            
    async def _deliver_to_subscriber(
        self,
        event: Event,
        subscription: Subscription
    ) -> None:
        """Deliver event to a single subscriber."""
        try:
            # Update subscription stats
            subscription.delivery_count += 1
            subscription.last_delivery = datetime.now(timezone.utc)
            
            # Call handler
            if asyncio.iscoroutinefunction(subscription.handler):
                await subscription.handler(event)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    subscription.handler,
                    event
                )
                
            self.logger.debug(
                f"Delivered event {event.id} to {subscription.subscriber}"
            )
            
        except Exception as e:
            subscription.error_count += 1
            subscription.last_error = str(e)
            
            self.logger.error(
                f"Failed to deliver event to {subscription.subscriber}: {e}"
            )
            
            # Retry logic for at-least-once delivery
            if event.delivery_mode == DeliveryMode.AT_LEAST_ONCE:
                # Re-queue with lower priority
                new_priority = min(event.priority.value + 1, EventPriority.LOW.value)
                event.priority = EventPriority(new_priority)
                await self._event_queue.put((new_priority, event))
                
    async def get_event_history(
        self,
        limit: int = 100,
        event_type: Optional[EventType] = None
    ) -> List[Event]:
        """Get recent event history."""
        history = list(self._event_history)
        
        if event_type:
            history = [e for e in history if e.type == event_type]
            
        return history[-limit:]
        
    async def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        stats = {
            "total_subscriptions": len(self._subscription_index),
            "active_subscriptions": sum(
                1 for s in self._subscription_index.values() if s.active
            ),
            "subscriptions_by_type": {}
        }
        
        for event_type, subs in self._subscriptions.items():
            stats["subscriptions_by_type"][event_type.value] = len(subs)
            
        return stats
        
    async def pause(self) -> None:
        """Pause event processing."""
        self._paused = True
        self.logger.info("Event processing paused")
        
    async def resume(self) -> None:
        """Resume event processing."""
        self._paused = False
        self.logger.info("Event processing resumed")
        
    async def shutdown(self) -> None:
        """Shutdown event bus."""
        self.logger.info("Shutting down event bus...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            await asyncio.gather(self._processing_task, return_exceptions=True)
            
        self.logger.info("Event bus shutdown complete")


class EventManager:
    """
    High-level event management interface.
    
    Provides simplified API for event publishing and subscription
    with additional features like event aggregation and correlation.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize event manager."""
        self.config = config or MCPAutomationConfig()
        self.logger = self._setup_logging()
        
        # Event bus
        self.event_bus = EventBus(config=config)
        
        # Event correlation
        self._correlations: Dict[str, List[Event]] = defaultdict(list)
        
        # Event aggregation
        self._aggregation_windows: Dict[str, Tuple[int, List[Event]]] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("mcp.event_manager")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def initialize(self) -> None:
        """Initialize event manager."""
        await self.event_bus.initialize()
        self.logger.info("Event manager initialized")
        
    async def publish(
        self,
        event: Union[Event, Dict[str, Any]],
        wait_for_delivery: bool = False
    ) -> str:
        """Publish an event."""
        # Convert dict to Event if needed
        if isinstance(event, dict):
            event = Event(**event)
            
        # Track correlation
        if event.correlation_id:
            self._correlations[event.correlation_id].append(event)
            
        # Publish event
        event_id = await self.event_bus.publish(event)
        
        # Wait for delivery if requested
        if wait_for_delivery:
            await asyncio.sleep(0.1)  # Give time for processing
            
        return event_id
        
    async def subscribe(
        self,
        event_type: Union[EventType, str],
        handler: Callable,
        **filter_kwargs
    ) -> str:
        """Subscribe to events with simplified API."""
        # Convert string to EventType
        if isinstance(event_type, str):
            event_type = EventType[event_type.upper()]
            
        # Build filter
        filter = EventFilter()
        if "sources" in filter_kwargs:
            filter.sources = set(filter_kwargs["sources"])
        if "priorities" in filter_kwargs:
            filter.priorities = set(filter_kwargs["priorities"])
        if "metadata" in filter_kwargs:
            filter.metadata_filters = filter_kwargs["metadata"]
            
        return await self.event_bus.subscribe(event_type, handler, filter)
        
    async def publish_system_event(
        self,
        action: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL
    ) -> str:
        """Publish a system event."""
        event = Event(
            type=EventType.SYSTEM,
            source="orchestrator",
            data={"action": action, **data},
            priority=priority
        )
        return await self.publish(event)
        
    async def publish_alert(
        self,
        message: str,
        severity: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Publish an alert event."""
        priority_map = {
            "low": EventPriority.LOW,
            "medium": EventPriority.NORMAL,
            "high": EventPriority.HIGH,
            "critical": EventPriority.CRITICAL
        }
        
        event = Event(
            type=EventType.ALERT,
            source="orchestrator",
            data={
                "message": message,
                "severity": severity,
                "details": details or {}
            },
            priority=priority_map.get(severity, EventPriority.NORMAL)
        )
        return await self.publish(event)
        
    async def get_correlated_events(self, correlation_id: str) -> List[Event]:
        """Get all events with the same correlation ID."""
        return self._correlations.get(correlation_id, [])
        
    async def aggregate_events(
        self,
        window_name: str,
        duration_seconds: int,
        event_type: Optional[EventType] = None
    ) -> List[Event]:
        """Aggregate events within a time window."""
        now = datetime.now(timezone.utc)
        
        if window_name in self._aggregation_windows:
            window_start, events = self._aggregation_windows[window_name]
            
            # Check if window expired
            if (now.timestamp() - window_start) > duration_seconds:
                # Start new window
                self._aggregation_windows[window_name] = (now.timestamp(), [])
                return events
            else:
                # Return current window events
                return events
        else:
            # Create new window
            self._aggregation_windows[window_name] = (now.timestamp(), [])
            return []
            
    async def wait_for_event(
        self,
        event_type: EventType,
        timeout: Optional[float] = None,
        filter: Optional[EventFilter] = None
    ) -> Optional[Event]:
        """Wait for a specific event."""
        received_event = None
        event_received = asyncio.Event()
        
        async def handler(event: Event):
            nonlocal received_event
            received_event = event
            event_received.set()
            
        # Subscribe temporarily
        sub_id = await self.event_bus.subscribe(event_type, handler, filter)
        
        try:
            # Wait for event
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event
        except asyncio.TimeoutError:
            return None
        finally:
            # Unsubscribe
            await self.event_bus.unsubscribe(sub_id)
            
    async def get_next_event(
        self,
        timeout: Optional[float] = None
    ) -> Optional[Event]:
        """Get next event from any source."""
        # This would typically be used by the orchestrator
        return await self.wait_for_event(
            EventType.CUSTOM,
            timeout=timeout
        )
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event system metrics."""
        return {
            **self.event_bus.metrics,
            "correlation_groups": len(self._correlations),
            "aggregation_windows": len(self._aggregation_windows)
        }
        
    async def shutdown(self) -> None:
        """Shutdown event manager."""
        await self.event_bus.shutdown()
        self.logger.info("Event manager shutdown complete")