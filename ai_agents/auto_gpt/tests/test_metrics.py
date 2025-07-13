#!/usr/bin/env python3.11
"""Tests for the metrics collection module."""

import pytest
import time
from typing import Dict, Any, List

    Metric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricRegistry,
    MetricCollector,
    MetricExporter,
)


@pytest.fixture
def metric_registry():
    """Create a test metric registry."""
    return MetricRegistry()


@pytest.fixture
def metric_collector(metric_registry):
    """Create a test metric collector."""
    return MetricCollector(metric_registry)


@pytest.fixture
def metric_exporter():
    """Create a test metric exporter."""
    return MetricExporter()


def test_counter_metric():
    """Test Counter metric functionality."""
    counter = Counter("test_counter", "Test counter metric")

    # Test initial value
    assert counter.value == 0

    # Test increment
    counter.inc()
    assert counter.value == 1

    # Test increment by value
    counter.inc(5)
    assert counter.value == 6

    # Test reset
    counter.reset()
    assert counter.value == 0


def test_gauge_metric():
    """Test Gauge metric functionality."""
    gauge = Gauge("test_gauge", "Test gauge metric")

    # Test initial value
    assert gauge.value == 0

    # Test set value
    gauge.set(42)
    assert gauge.value == 42

    # Test increment
    gauge.inc(10)
    assert gauge.value == 52

    # Test decrement
    gauge.dec(5)
    assert gauge.value == 47

    # Test reset
    gauge.reset()
    assert gauge.value == 0


def test_histogram_metric():
    """Test Histogram metric functionality."""
    histogram = Histogram("test_histogram", "Test histogram metric")

    # Test initial state
    assert histogram.count == 0
    assert histogram.sum == 0
    assert len(histogram.buckets) == 0

    # Test observe values
    values = [1, 2, 3, 4, 5]
    for value in values:
        histogram.observe(value)

    assert histogram.count == 5
    assert histogram.sum == 15
    assert len(histogram.buckets) == 5

    # Test bucket calculation
    buckets = histogram.calculate_buckets()
    assert len(buckets) > 0
    assert all(count >= 0 for count in buckets.values())

    # Test reset
    histogram.reset()
    assert histogram.count == 0
    assert histogram.sum == 0
    assert len(histogram.buckets) == 0


def test_summary_metric():
    """Test Summary metric functionality."""
    summary = Summary("test_summary", "Test summary metric")

    # Test initial state
    assert summary.count == 0
    assert summary.sum == 0
    assert len(summary.quantiles) == 0

    # Test observe values
    values = [1, 2, 3, 4, 5]
    for value in values:
        summary.observe(value)

    assert summary.count == 5
    assert summary.sum == 15
    assert len(summary.quantiles) == 5

    # Test quantile calculation
    quantiles = summary.calculate_quantiles([0.5, 0.9, 0.95])
    assert len(quantiles) == 3
    assert all(value >= 0 for value in quantiles.values())

    # Test reset
    summary.reset()
    assert summary.count == 0
    assert summary.sum == 0
    assert len(summary.quantiles) == 0


def test_metric_registry(metric_registry):
    """Test MetricRegistry functionality."""
    # Test metric registration
    counter = Counter("test_counter", "Test counter")
    metric_registry.register(counter)
    assert counter in metric_registry.metrics

    # Test duplicate registration
    with pytest.raises(ValueError):
        metric_registry.register(counter)

    # Test metric retrieval
    retrieved = metric_registry.get_metric("test_counter")
    assert retrieved == counter

    # Test non-existent metric
    assert metric_registry.get_metric("non_existent") is None

    # Test metric unregistration
    metric_registry.unregister("test_counter")
    assert counter not in metric_registry.metrics


def test_metric_collector(metric_collector):
    """Test MetricCollector functionality."""
    # Test metric collection
    counter = Counter("test_counter", "Test counter")
    metric_collector.registry.register(counter)

    # Record some metrics
    counter.inc()
    counter.inc(5)

    # Collect metrics
    metrics = metric_collector.collect()
    assert len(metrics) == 1
    assert metrics[0].name == "test_counter"
    assert metrics[0].value == 6

    # Test collection with labels
    gauge = Gauge("test_gauge", "Test gauge", labels=["type"])
    metric_collector.registry.register(gauge)

    gauge.labels(type="test").set(42)
    metrics = metric_collector.collect()
    assert len(metrics) == 2
    assert any(m.name == "test_gauge" and m.value == 42 for m in metrics)


def test_metric_exporter(metric_exporter, metric_collector):
    """Test MetricExporter functionality."""
    # Setup test metrics
    counter = Counter("test_counter", "Test counter")
    metric_collector.registry.register(counter)
    counter.inc(42)

    # Test export to string
    exported = metric_exporter.export(metric_collector.collect())
    assert "test_counter" in exported
    assert "42" in exported

    # Test export to file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        metric_exporter.export_to_file(metric_collector.collect(), f.name)
        with open(f.name) as f:
            content = f.read()
            assert "test_counter" in content
            assert "42" in content
    os.unlink(f.name)


def test_metric_timing():
    """Test metric timing functionality."""
    histogram = Histogram("test_timing", "Test timing metric")

    # Test timing context manager
    with histogram.time():
        time.sleep(0.1)

    assert histogram.count == 1
    assert histogram.sum >= 0.1

    # Test timing decorator
    @histogram.time()
    def test_function():
        time.sleep(0.1)

    test_function()
    assert histogram.count == 2
    assert histogram.sum >= 0.2


def test_metric_labels():
    """Test metric labels functionality."""
    # Test counter with labels
    counter = Counter("test_counter", "Test counter", labels=["type", "status"])

    # Test label creation
    labeled_counter = counter.labels(type="test", status="success")
    assert labeled_counter.labels == {"type": "test", "status": "success"}

    # Test metric recording with labels
    labeled_counter.inc()
    assert labeled_counter.value == 1

    # Test different label combinations
    other_counter = counter.labels(type="other", status="error")
    other_counter.inc(2)
    assert other_counter.value == 2
    assert labeled_counter.value == 1


def test_metric_aggregation():
    """Test metric aggregation functionality."""
    # Test histogram aggregation
    histogram = Histogram("test_histogram", "Test histogram")
    values = [1, 2, 3, 4, 5]
    for value in values:
        histogram.observe(value)

    # Test basic statistics
    assert histogram.count == 5
    assert histogram.sum == 15
    assert histogram.avg == 3

    # Test bucket aggregation
    buckets = histogram.calculate_buckets()
    assert len(buckets) > 0
    assert sum(buckets.values()) == 5

    # Test summary aggregation
    summary = Summary("test_summary", "Test summary")
    for value in values:
        summary.observe(value)

    # Test quantile calculation
    quantiles = summary.calculate_quantiles([0.5, 0.9])
    assert len(quantiles) == 2
    assert all(0 <= value <= 5 for value in quantiles.values())
