#!/usr/bin/env python3
"""
MCP Automation SLA Monitor
Service Level Agreement tracking and reporting system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SLIType(Enum):
    """Types of Service Level Indicators"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUALITY = "quality"
    CAPACITY = "capacity"


class ComplianceStatus(Enum):
    """SLA compliance status"""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class SLI:
    """Service Level Indicator"""
    name: str
    type: SLIType
    description: str
    measurement: str  # How to measure this SLI
    unit: str  # Unit of measurement
    aggregation: str  # sum, avg, max, min, percentile
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLO:
    """Service Level Objective"""
    name: str
    sli: SLI
    target: float
    warning_threshold: float  # Threshold for "at risk" status
    time_window: timedelta
    description: str
    consequences: str  # What happens if violated
    enabled: bool = True


@dataclass
class SLAMeasurement:
    """Individual SLA measurement"""
    slo_name: str
    timestamp: datetime
    value: float
    target: float
    compliance_status: ComplianceStatus
    error_budget_consumed: float  # Percentage of error budget used
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAReport:
    """SLA compliance report"""
    period_start: datetime
    period_end: datetime
    slo_reports: Dict[str, 'SLOReport'] = field(default_factory=dict)
    overall_compliance: float = 0.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SLOReport:
    """Individual SLO compliance report"""
    slo_name: str
    compliance_percentage: float
    measurements_total: int
    measurements_compliant: int
    measurements_violated: int
    average_value: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    error_budget_remaining: float
    trend: str  # improving, stable, degrading
    violations: List[SLAMeasurement] = field(default_factory=list)


class SLAMonitor:
    """SLA monitoring and compliance tracking system"""
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 history_retention_days: int = 90):
        """
        Initialize SLA monitor
        
        Args:
            config_path: Path to SLA configuration file
            history_retention_days: Days to retain measurement history
        """
        self.config_path = Path(config_path) if config_path else None
        self.history_retention_days = history_retention_days
        
        # SLO definitions
        self.slos: Dict[str, SLO] = {}
        
        # Measurement storage
        self.measurements: Dict[str, List[SLAMeasurement]] = {}
        self.current_status: Dict[str, ComplianceStatus] = {}
        
        # Load configuration
        self._load_slos()
        
        # Statistics
        self.stats = {
            'measurements_total': 0,
            'violations_total': 0,
            'reports_generated': 0
        }
        
    def _load_slos(self):
        """Load SLO definitions"""
        # Default SLOs for MCP automation
        default_slos = [
            SLO(
                name="mcp_availability",
                sli=SLI(
                    name="mcp_server_availability",
                    type=SLIType.AVAILABILITY,
                    description="MCP server availability",
                    measurement="mcp_server_up",
                    unit="ratio",
                    aggregation="avg"
                ),
                target=0.999,  # 99.9% availability
                warning_threshold=0.995,
                time_window=timedelta(days=30),
                description="MCP servers must maintain 99.9% availability",
                consequences="Automated failover and incident escalation"
            ),
            SLO(
                name="api_latency_p95",
                sli=SLI(
                    name="api_response_time",
                    type=SLIType.LATENCY,
                    description="API response time 95th percentile",
                    measurement="mcp_server_latency_seconds",
                    unit="seconds",
                    aggregation="percentile_95"
                ),
                target=0.2,  # 200ms
                warning_threshold=0.15,
                time_window=timedelta(hours=24),
                description="95% of API requests must complete within 200ms",
                consequences="Performance optimization required"
            ),
            SLO(
                name="automation_success_rate",
                sli=SLI(
                    name="automation_success",
                    type=SLIType.QUALITY,
                    description="Automation workflow success rate",
                    measurement="automation_success_rate",
                    unit="ratio",
                    aggregation="avg"
                ),
                target=0.98,  # 98% success rate
                warning_threshold=0.95,
                time_window=timedelta(days=7),
                description="Automation workflows must maintain 98% success rate",
                consequences="Manual intervention and root cause analysis"
            ),
            SLO(
                name="error_rate",
                sli=SLI(
                    name="system_error_rate",
                    type=SLIType.ERROR_RATE,
                    description="System error rate",
                    measurement="error_rate",
                    unit="ratio",
                    aggregation="avg"
                ),
                target=0.01,  # 1% error rate max
                warning_threshold=0.02,
                time_window=timedelta(hours=1),
                description="System error rate must stay below 1%",
                consequences="Immediate investigation and remediation"
            ),
            SLO(
                name="data_freshness",
                sli=SLI(
                    name="data_update_lag",
                    type=SLIType.LATENCY,
                    description="Data update lag time",
                    measurement="data_lag_seconds",
                    unit="seconds",
                    aggregation="max"
                ),
                target=60,  # 60 seconds max lag
                warning_threshold=30,
                time_window=timedelta(hours=1),
                description="Data must be updated within 60 seconds",
                consequences="Data pipeline optimization required"
            ),
            SLO(
                name="resource_utilization",
                sli=SLI(
                    name="cpu_memory_utilization",
                    type=SLIType.CAPACITY,
                    description="System resource utilization",
                    measurement="resource_utilization_percent",
                    unit="percent",
                    aggregation="percentile_95"
                ),
                target=80,  # 80% max utilization
                warning_threshold=70,
                time_window=timedelta(hours=4),
                description="Resource utilization must stay below 80%",
                consequences="Capacity planning and scaling required"
            )
        ]
        
        # Load SLOs
        for slo in default_slos:
            self.slos[slo.name] = slo
            self.measurements[slo.name] = []
            self.current_status[slo.name] = ComplianceStatus.UNKNOWN
            
        # Load custom SLOs from config if available
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Process custom SLOs
                    for slo_config in config.get('slos', []):
                        self._add_custom_slo(slo_config)
            except Exception as e:
                logger.error(f"Failed to load SLO config: {e}")
                
    def _add_custom_slo(self, slo_config: Dict[str, Any]):
        """Add custom SLO from configuration"""
        try:
            sli = SLI(
                name=slo_config['sli']['name'],
                type=SLIType(slo_config['sli']['type']),
                description=slo_config['sli'].get('description', ''),
                measurement=slo_config['sli']['measurement'],
                unit=slo_config['sli'].get('unit', ''),
                aggregation=slo_config['sli'].get('aggregation', 'avg')
            )
            
            slo = SLO(
                name=slo_config['name'],
                sli=sli,
                target=float(slo_config['target']),
                warning_threshold=float(slo_config.get('warning_threshold', slo_config['target'] * 0.95)),
                time_window=timedelta(**slo_config.get('time_window', {'hours': 24})),
                description=slo_config.get('description', ''),
                consequences=slo_config.get('consequences', ''),
                enabled=slo_config.get('enabled', True)
            )
            
            self.slos[slo.name] = slo
            self.measurements[slo.name] = []
            self.current_status[slo.name] = ComplianceStatus.UNKNOWN
            
            logger.info(f"Added custom SLO: {slo.name}")
            
        except Exception as e:
            logger.error(f"Failed to add custom SLO: {e}")
            
    def record_measurement(self,
                          slo_name: str,
                          value: float,
                          timestamp: Optional[datetime] = None) -> SLAMeasurement:
        """
        Record an SLA measurement
        
        Args:
            slo_name: Name of the SLO
            value: Measured value
            timestamp: Measurement timestamp (default: now)
            
        Returns:
            SLAMeasurement object
        """
        if slo_name not in self.slos:
            raise ValueError(f"Unknown SLO: {slo_name}")
            
        slo = self.slos[slo_name]
        timestamp = timestamp or datetime.now()
        
        # Determine compliance status
        if slo.sli.type == SLIType.ERROR_RATE:
            # For error rate, lower is better
            if value <= slo.target:
                status = ComplianceStatus.COMPLIANT
            elif value <= slo.warning_threshold:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.VIOLATED
        elif slo.sli.type in [SLIType.LATENCY, SLIType.CAPACITY]:
            # For latency and capacity, lower is better
            if value <= slo.target:
                status = ComplianceStatus.COMPLIANT
            elif value <= slo.warning_threshold:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.VIOLATED
        else:
            # For availability, throughput, quality - higher is better
            if value >= slo.target:
                status = ComplianceStatus.COMPLIANT
            elif value >= slo.warning_threshold:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.VIOLATED
                
        # Calculate error budget consumption
        if slo.sli.type in [SLIType.ERROR_RATE, SLIType.LATENCY, SLIType.CAPACITY]:
            error_budget_consumed = (value / slo.target) * 100 if slo.target > 0 else 100
        else:
            error_budget_consumed = ((slo.target - value) / (slo.target)) * 100 if slo.target > 0 else 0
            
        error_budget_consumed = max(0, min(100, error_budget_consumed))
        
        measurement = SLAMeasurement(
            slo_name=slo_name,
            timestamp=timestamp,
            value=value,
            target=slo.target,
            compliance_status=status,
            error_budget_consumed=error_budget_consumed
        )
        
        # Store measurement
        self.measurements[slo_name].append(measurement)
        self.current_status[slo_name] = status
        
        # Update statistics
        self.stats['measurements_total'] += 1
        if status == ComplianceStatus.VIOLATED:
            self.stats['violations_total'] += 1
            
        # Clean old measurements
        self._clean_old_measurements(slo_name)
        
        logger.debug(f"Recorded measurement for {slo_name}: {value} ({status.value})")
        
        return measurement
        
    def _clean_old_measurements(self, slo_name: str):
        """Remove old measurements beyond retention period"""
        cutoff_time = datetime.now() - timedelta(days=self.history_retention_days)
        self.measurements[slo_name] = [
            m for m in self.measurements[slo_name]
            if m.timestamp > cutoff_time
        ]
        
    def calculate_compliance(self,
                           slo_name: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> float:
        """
        Calculate compliance percentage for an SLO
        
        Args:
            slo_name: Name of the SLO
            start_time: Start of period (default: based on SLO time window)
            end_time: End of period (default: now)
            
        Returns:
            Compliance percentage (0-100)
        """
        if slo_name not in self.slos:
            raise ValueError(f"Unknown SLO: {slo_name}")
            
        slo = self.slos[slo_name]
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - slo.time_window)
        
        # Filter measurements in time window
        measurements = [
            m for m in self.measurements[slo_name]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not measurements:
            return 0.0
            
        compliant_count = sum(1 for m in measurements if m.compliance_status == ComplianceStatus.COMPLIANT)
        compliance_percentage = (compliant_count / len(measurements)) * 100
        
        return compliance_percentage
        
    def generate_slo_report(self,
                          slo_name: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> SLOReport:
        """
        Generate detailed SLO compliance report
        
        Args:
            slo_name: Name of the SLO
            start_time: Start of reporting period
            end_time: End of reporting period
            
        Returns:
            SLOReport object
        """
        if slo_name not in self.slos:
            raise ValueError(f"Unknown SLO: {slo_name}")
            
        slo = self.slos[slo_name]
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - slo.time_window)
        
        # Filter measurements
        measurements = [
            m for m in self.measurements[slo_name]
            if start_time <= m.timestamp <= end_time
        ]
        
        if not measurements:
            return SLOReport(
                slo_name=slo_name,
                compliance_percentage=0.0,
                measurements_total=0,
                measurements_compliant=0,
                measurements_violated=0,
                average_value=0.0,
                min_value=0.0,
                max_value=0.0,
                percentile_95=0.0,
                percentile_99=0.0,
                error_budget_remaining=0.0,
                trend="unknown"
            )
            
        # Calculate statistics
        values = [m.value for m in measurements]
        compliant = [m for m in measurements if m.compliance_status == ComplianceStatus.COMPLIANT]
        violated = [m for m in measurements if m.compliance_status == ComplianceStatus.VIOLATED]
        
        # Calculate percentiles
        sorted_values = sorted(values)
        p95_index = int(len(sorted_values) * 0.95)
        p99_index = int(len(sorted_values) * 0.99)
        
        # Determine trend
        if len(measurements) > 10:
            first_half = measurements[:len(measurements)//2]
            second_half = measurements[len(measurements)//2:]
            
            first_compliance = sum(1 for m in first_half if m.compliance_status == ComplianceStatus.COMPLIANT) / len(first_half)
            second_compliance = sum(1 for m in second_half if m.compliance_status == ComplianceStatus.COMPLIANT) / len(second_half)
            
            if second_compliance > first_compliance + 0.05:
                trend = "improving"
            elif second_compliance < first_compliance - 0.05:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "unknown"
            
        # Calculate error budget
        error_budget_consumed = statistics.mean([m.error_budget_consumed for m in measurements])
        error_budget_remaining = max(0, 100 - error_budget_consumed)
        
        report = SLOReport(
            slo_name=slo_name,
            compliance_percentage=(len(compliant) / len(measurements)) * 100,
            measurements_total=len(measurements),
            measurements_compliant=len(compliant),
            measurements_violated=len(violated),
            average_value=statistics.mean(values),
            min_value=min(values),
            max_value=max(values),
            percentile_95=sorted_values[p95_index] if p95_index < len(sorted_values) else sorted_values[-1],
            percentile_99=sorted_values[p99_index] if p99_index < len(sorted_values) else sorted_values[-1],
            error_budget_remaining=error_budget_remaining,
            trend=trend,
            violations=violated[:10]  # Include up to 10 most recent violations
        )
        
        return report
        
    def generate_sla_report(self,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> SLAReport:
        """
        Generate comprehensive SLA compliance report
        
        Args:
            start_time: Start of reporting period
            end_time: End of reporting period
            
        Returns:
            SLAReport object
        """
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=30))
        
        report = SLAReport(
            period_start=start_time,
            period_end=end_time
        )
        
        # Generate individual SLO reports
        total_compliance = 0
        enabled_slos = 0
        
        for slo_name, slo in self.slos.items():
            if not slo.enabled:
                continue
                
            slo_report = self.generate_slo_report(slo_name, start_time, end_time)
            report.slo_reports[slo_name] = slo_report
            
            total_compliance += slo_report.compliance_percentage
            enabled_slos += 1
            
            # Track violations
            for violation in slo_report.violations:
                report.violations.append({
                    'slo': slo_name,
                    'timestamp': violation.timestamp.isoformat(),
                    'value': violation.value,
                    'target': violation.target,
                    'impact': slo.consequences
                })
                
        # Calculate overall compliance
        if enabled_slos > 0:
            report.overall_compliance = total_compliance / enabled_slos
        else:
            report.overall_compliance = 0.0
            
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        self.stats['reports_generated'] += 1
        
        return report
        
    def _generate_recommendations(self, report: SLAReport) -> List[str]:
        """Generate recommendations based on SLA report"""
        recommendations = []
        
        for slo_name, slo_report in report.slo_reports.items():
            slo = self.slos[slo_name]
            
            # Check compliance
            if slo_report.compliance_percentage < 90:
                recommendations.append(
                    f"CRITICAL: {slo_name} compliance is {slo_report.compliance_percentage:.1f}%. "
                    f"Immediate action required: {slo.consequences}"
                )
            elif slo_report.compliance_percentage < 95:
                recommendations.append(
                    f"WARNING: {slo_name} compliance is {slo_report.compliance_percentage:.1f}%. "
                    f"Consider preventive measures."
                )
                
            # Check error budget
            if slo_report.error_budget_remaining < 20:
                recommendations.append(
                    f"Low error budget for {slo_name} ({slo_report.error_budget_remaining:.1f}% remaining). "
                    f"Freeze non-critical changes."
                )
                
            # Check trends
            if slo_report.trend == "degrading":
                recommendations.append(
                    f"Degrading trend detected for {slo_name}. Investigate root cause."
                )
                
            # Specific recommendations by SLI type
            if slo.sli.type == SLIType.LATENCY and slo_report.percentile_95 > slo.target:
                recommendations.append(
                    f"Latency SLO {slo_name} P95 ({slo_report.percentile_95:.2f}) exceeds target. "
                    f"Consider performance optimization."
                )
            elif slo.sli.type == SLIType.CAPACITY and slo_report.average_value > 70:
                recommendations.append(
                    f"Capacity utilization for {slo_name} is high ({slo_report.average_value:.1f}%). "
                    f"Plan for scaling."
                )
                
        return recommendations
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get current SLA status summary"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'slos': {},
            'overall_health': 'healthy',
            'at_risk_count': 0,
            'violated_count': 0
        }
        
        for slo_name, compliance_status in self.current_status.items():
            if not self.slos[slo_name].enabled:
                continue
                
            status['slos'][slo_name] = {
                'status': compliance_status.value,
                'compliance': self.calculate_compliance(slo_name),
                'last_measurement': self.measurements[slo_name][-1].timestamp.isoformat() 
                    if self.measurements[slo_name] else None
            }
            
            if compliance_status == ComplianceStatus.AT_RISK:
                status['at_risk_count'] += 1
            elif compliance_status == ComplianceStatus.VIOLATED:
                status['violated_count'] += 1
                
        # Determine overall health
        if status['violated_count'] > 0:
            status['overall_health'] = 'unhealthy'
        elif status['at_risk_count'] > 0:
            status['overall_health'] = 'at_risk'
            
        return status
        
    def export_report(self, report: SLAReport, output_path: str):
        """Export SLA report to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to dictionary
        report_dict = {
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'overall_compliance': report.overall_compliance,
            'generated_at': report.generated_at.isoformat(),
            'slo_reports': {},
            'violations': report.violations,
            'recommendations': report.recommendations
        }
        
        for slo_name, slo_report in report.slo_reports.items():
            report_dict['slo_reports'][slo_name] = {
                'compliance_percentage': slo_report.compliance_percentage,
                'measurements_total': slo_report.measurements_total,
                'measurements_compliant': slo_report.measurements_compliant,
                'measurements_violated': slo_report.measurements_violated,
                'average_value': slo_report.average_value,
                'min_value': slo_report.min_value,
                'max_value': slo_report.max_value,
                'percentile_95': slo_report.percentile_95,
                'percentile_99': slo_report.percentile_99,
                'error_budget_remaining': slo_report.error_budget_remaining,
                'trend': slo_report.trend
            }
            
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
            
        logger.info(f"SLA report exported to: {output_file}")


async def main():
    """Main function for testing"""
    monitor = SLAMonitor()
    
    # Simulate measurements
    import random
    
    # Record some measurements
    for _ in range(100):
        # Availability measurements
        monitor.record_measurement(
            'mcp_availability',
            random.uniform(0.995, 1.0)  # Mostly compliant
        )
        
        # Latency measurements
        monitor.record_measurement(
            'api_latency_p95',
            random.uniform(0.05, 0.25)  # Some violations
        )
        
        # Success rate measurements
        monitor.record_measurement(
            'automation_success_rate',
            random.uniform(0.96, 1.0)  # Mostly compliant
        )
        
    # Generate report
    report = monitor.generate_sla_report()
    
    # Print summary
    print(f"Overall Compliance: {report.overall_compliance:.2f}%")
    print(f"Violations: {len(report.violations)}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
        
    # Get current status
    status = monitor.get_current_status()
    print(f"\nCurrent Status: {status['overall_health']}")
    print(f"At Risk: {status['at_risk_count']}, Violated: {status['violated_count']}")
    
    # Export report
    monitor.export_report(report, "/tmp/sla_report.json")


if __name__ == "__main__":
    asyncio.run(main())