"""
Data Quality Monitor
===================

Comprehensive data quality monitoring system that assesses and tracks
data quality across multiple dimensions including completeness, accuracy,
consistency, timeliness, and validity.
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import statistics
import hashlib


class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"


class QualityIssueType(Enum):
    """Types of data quality issues"""
    MISSING_VALUES = "missing_values"
    INVALID_FORMAT = "invalid_format"
    DUPLICATE_RECORDS = "duplicate_records"
    OUTLIER_VALUES = "outlier_values"
    INCONSISTENT_VALUES = "inconsistent_values"
    STALE_DATA = "stale_data"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass
class QualityRule:
    """Defines a data quality rule"""
    id: str
    name: str
    description: str
    dimension: QualityDimension
    
    # Rule configuration
    rule_type: str  # regex, range, completeness, uniqueness, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Applicability
    data_types: Optional[Set[str]] = None
    columns: Optional[Set[str]] = None
    tables: Optional[Set[str]] = None
    
    # Thresholds
    error_threshold: float = 0.0  # Percentage of failures that constitute an error
    warning_threshold: float = 0.05  # Percentage of failures that constitute a warning
    
    # Metadata
    severity: str = "medium"  # low, medium, high, critical
    remediation_suggestion: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    id: str
    rule_id: str
    data_id: str
    issue_type: QualityIssueType
    dimension: QualityDimension
    
    # Issue details
    description: str
    severity: str
    failure_count: int = 0
    total_records: int = 0
    failure_percentage: float = 0.0
    
    # Examples and context
    sample_failures: List[Any] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    
    # Resolution tracking
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Results of a data quality assessment"""
    data_id: str
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Overall scores
    overall_score: float = 0.0
    dimension_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    
    # Issues found
    issues: List[QualityIssue] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    # Data statistics
    total_records: int = 0
    total_columns: int = 0
    null_percentage: float = 0.0
    duplicate_percentage: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    assessment_duration_ms: Optional[int] = None
    rules_evaluated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system that continuously
    assesses and tracks data quality across multiple dimensions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("quality_monitor")
        
        # Quality rules
        self.rules: Dict[str, QualityRule] = {}
        self.issues: Dict[str, QualityIssue] = {}
        
        # Configuration
        self.batch_size = self.config.get('batch_size', 1000)
        self.assessment_interval_hours = self.config.get('assessment_interval_hours', 6)
        self.max_sample_failures = self.config.get('max_sample_failures', 10)
        
        # Quality thresholds
        self.default_quality_threshold = self.config.get('default_quality_threshold', 0.8)
        self.critical_quality_threshold = self.config.get('critical_quality_threshold', 0.5)
        
        # Statistics
        self.stats = {
            "assessments_completed": 0,
            "issues_detected": 0,
            "issues_resolved": 0,
            "avg_quality_score": 0.0
        }
        
        # Initialize default quality rules
        self._initialize_default_rules()
    
    async def initialize(self) -> bool:
        """Initialize the quality monitor"""
        try:
            self.logger.info("Initializing data quality monitor")
            
            # Validate configuration
            if self.default_quality_threshold < 0 or self.default_quality_threshold > 1:
                raise ValueError("Quality threshold must be between 0 and 1")
            
            self.logger.info(f"Quality monitor initialized with {len(self.rules)} rules")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quality monitor: {e}")
            return False
    
    def _initialize_default_rules(self):
        """Initialize default data quality rules"""
        
        # Completeness rules
        self.add_rule(QualityRule(
            id="completeness_required_fields",
            name="Required Fields Completeness",
            description="Check that required fields are not null or empty",
            dimension=QualityDimension.COMPLETENESS,
            rule_type="completeness",
            parameters={"required_fields": ["id", "name", "created_at"]},
            error_threshold=0.0,
            warning_threshold=0.01,
            severity="high",
            remediation_suggestion="Ensure data pipeline populates all required fields"
        ))
        
        # Validity rules
        self.add_rule(QualityRule(
            id="email_format_validation",
            name="Email Format Validation",
            description="Validate email addresses follow correct format",
            dimension=QualityDimension.VALIDITY,
            rule_type="regex",
            parameters={"pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            columns={"email", "email_address", "contact_email"},
            error_threshold=0.05,
            warning_threshold=0.02,
            severity="medium",
            remediation_suggestion="Implement email validation at data entry points"
        ))
        
        self.add_rule(QualityRule(
            id="phone_format_validation",
            name="Phone Number Format",
            description="Validate phone numbers follow standard formats",
            dimension=QualityDimension.VALIDITY,
            rule_type="regex",
            parameters={"pattern": r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'},
            columns={"phone", "phone_number", "mobile"},
            error_threshold=0.1,
            warning_threshold=0.05,
            severity="low",
            remediation_suggestion="Standardize phone number format during data ingestion"
        ))
        
        # Uniqueness rules
        self.add_rule(QualityRule(
            id="primary_key_uniqueness",
            name="Primary Key Uniqueness",
            description="Ensure primary key values are unique",
            dimension=QualityDimension.UNIQUENESS,
            rule_type="uniqueness",
            parameters={"key_columns": ["id"]},
            error_threshold=0.0,
            warning_threshold=0.0,
            severity="critical",
            remediation_suggestion="Investigate data ingestion process for duplicate key generation"
        ))
        
        # Consistency rules
        self.add_rule(QualityRule(
            id="date_consistency",
            name="Date Consistency",
            description="Check that dates are logically consistent",
            dimension=QualityDimension.CONSISTENCY,
            rule_type="date_consistency",
            parameters={"date_order_checks": [("created_at", "updated_at")]},
            error_threshold=0.01,
            warning_threshold=0.005,
            severity="medium",
            remediation_suggestion="Review date assignment logic in applications"
        ))
        
        # Timeliness rules
        self.add_rule(QualityRule(
            id="data_freshness",
            name="Data Freshness",
            description="Check that data is not stale",
            dimension=QualityDimension.TIMELINESS,
            rule_type="freshness",
            parameters={"max_age_hours": 24},
            error_threshold=0.1,
            warning_threshold=0.05,
            severity="medium",
            remediation_suggestion="Review data refresh schedules"
        ))
        
        # Accuracy rules
        self.add_rule(QualityRule(
            id="numeric_range_validation",
            name="Numeric Range Validation",
            description="Validate numeric values are within expected ranges",
            dimension=QualityDimension.ACCURACY,
            rule_type="numeric_range",
            parameters={"age": {"min": 0, "max": 150}, "percentage": {"min": 0, "max": 100}},
            error_threshold=0.02,
            warning_threshold=0.01,
            severity="medium",
            remediation_suggestion="Add range validation to data entry forms and APIs"
        ))
    
    def add_rule(self, rule: QualityRule):
        """Add a quality rule"""
        self.rules[rule.id] = rule
        self.logger.debug(f"Added quality rule: {rule.id}")
    
    async def assess_quality(self, data_id: str, content: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Assess data quality and return overall quality score
        
        Args:
            data_id: Unique identifier for the data
            content: Data content to assess
            metadata: Additional metadata about the data
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        try:
            start_time = datetime.utcnow()
            
            # Perform comprehensive quality assessment
            assessment = await self._perform_quality_assessment(data_id, content, metadata)
            
            # Calculate assessment duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            assessment.assessment_duration_ms = int(duration)
            
            # Update statistics
            self.stats["assessments_completed"] += 1
            self.stats["issues_detected"] += assessment.total_issues
            
            # Update average quality score
            current_avg = self.stats["avg_quality_score"]
            count = self.stats["assessments_completed"]
            self.stats["avg_quality_score"] = ((current_avg * (count - 1)) + assessment.overall_score) / count
            
            self.logger.debug(f"Quality assessment for {data_id}: score={assessment.overall_score:.3f}, issues={assessment.total_issues}")
            
            return assessment.overall_score
            
        except Exception as e:
            self.logger.error(f"Error assessing quality for {data_id}: {e}")
            return 0.0
    
    async def _perform_quality_assessment(self, data_id: str, content: str,
                                        metadata: Optional[Dict[str, Any]]) -> QualityAssessment:
        """Perform comprehensive quality assessment"""
        
        assessment = QualityAssessment(data_id=data_id)
        
        # Parse content (simplified - in practice would handle various data formats)
        parsed_data = self._parse_content(content, metadata)
        
        if not parsed_data:
            assessment.overall_score = 0.0
            assessment.recommendations.append("Unable to parse data content")
            return assessment
        
        assessment.total_records = len(parsed_data.get('records', []))
        assessment.total_columns = len(parsed_data.get('columns', []))
        
        # Initialize dimension scores
        dimension_scores = {}
        
        # Evaluate each applicable rule
        for rule in self.rules.values():
            if not rule.is_active:
                continue
                
            if not self._rule_applies_to_data(rule, metadata, parsed_data):
                continue
            
            assessment.rules_evaluated += 1
            
            # Evaluate rule
            rule_result = await self._evaluate_quality_rule(rule, data_id, parsed_data, metadata)
            
            # Update dimension score
            dimension = rule.dimension
            if dimension not in dimension_scores:
                dimension_scores[dimension] = []
            dimension_scores[dimension].append(rule_result['score'])
            
            # Check for issues
            if rule_result['has_issues']:
                issue = self._create_quality_issue(rule, data_id, rule_result)
                assessment.issues.append(issue)
                self.issues[issue.id] = issue
                
                # Count by severity
                if issue.severity == "critical":
                    assessment.critical_issues += 1
                elif issue.severity == "high":
                    assessment.high_issues += 1
                elif issue.severity == "medium":
                    assessment.medium_issues += 1
                else:
                    assessment.low_issues += 1
        
        assessment.total_issues = len(assessment.issues)
        
        # Calculate dimension scores
        for dimension, scores in dimension_scores.items():
            if scores:
                assessment.dimension_scores[dimension] = statistics.mean(scores)
        
        # Calculate overall score
        if assessment.dimension_scores:
            assessment.overall_score = statistics.mean(assessment.dimension_scores.values())
        else:
            assessment.overall_score = 1.0  # No rules applied
        
        # Calculate basic statistics
        if parsed_data.get('records'):
            assessment.null_percentage = self._calculate_null_percentage(parsed_data['records'])
            assessment.duplicate_percentage = self._calculate_duplicate_percentage(parsed_data['records'])
        
        # Generate recommendations
        assessment.recommendations = self._generate_quality_recommendations(assessment)
        
        return assessment
    
    def _parse_content(self, content: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse content into structured format for quality assessment"""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return {"records": data, "columns": list(data[0].keys()) if data and isinstance(data[0], dict) else []}
                    elif isinstance(data, dict):
                        return {"records": [data], "columns": list(data.keys())}
                except json.JSONDecodeError:
                    pass
            
            # Try to parse as CSV-like content
            lines = content.strip().split('\n')
            if len(lines) > 1:
                # Assume first line is headers
                headers = [h.strip() for h in lines[0].split(',')]
                records = []
                
                for line in lines[1:]:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(headers):
                        record = dict(zip(headers, values))
                        records.append(record)
                
                return {"records": records, "columns": headers}
            
            # Fallback: treat as single text record
            return {
                "records": [{"content": content}],
                "columns": ["content"]
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing content: {e}")
            return {}
    
    def _rule_applies_to_data(self, rule: QualityRule, metadata: Optional[Dict[str, Any]],
                            parsed_data: Dict[str, Any]) -> bool:
        """Check if a quality rule applies to the given data"""
        
        # Check table filters
        if rule.tables and metadata:
            table_name = metadata.get('table_name', '').lower()
            if table_name and not any(table.lower() in table_name for table in rule.tables):
                return False
        
        # Check column filters
        if rule.columns:
            data_columns = set(col.lower() for col in parsed_data.get('columns', []))
            rule_columns = set(col.lower() for col in rule.columns)
            if not data_columns.intersection(rule_columns):
                return False
        
        # Check data type filters
        if rule.data_types and metadata:
            data_type = metadata.get('data_type', '').lower()
            if data_type and data_type not in [dt.lower() for dt in rule.data_types]:
                return False
        
        return True
    
    async def _evaluate_quality_rule(self, rule: QualityRule, data_id: str,
                                   parsed_data: Dict[str, Any],
                                   metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a specific quality rule"""
        
        result = {
            "rule_id": rule.id,
            "score": 1.0,
            "has_issues": False,
            "failure_count": 0,
            "total_count": 0,
            "failure_percentage": 0.0,
            "sample_failures": [],
            "details": {}
        }
        
        try:
            records = parsed_data.get('records', [])
            if not records:
                return result
            
            result["total_count"] = len(records)
            
            # Evaluate based on rule type
            if rule.rule_type == "completeness":
                result = await self._evaluate_completeness_rule(rule, records, result)
            elif rule.rule_type == "regex":
                result = await self._evaluate_regex_rule(rule, records, result)
            elif rule.rule_type == "uniqueness":
                result = await self._evaluate_uniqueness_rule(rule, records, result)
            elif rule.rule_type == "numeric_range":
                result = await self._evaluate_numeric_range_rule(rule, records, result)
            elif rule.rule_type == "date_consistency":
                result = await self._evaluate_date_consistency_rule(rule, records, result)
            elif rule.rule_type == "freshness":
                result = await self._evaluate_freshness_rule(rule, records, metadata, result)
            else:
                # Generic rule evaluation
                result = await self._evaluate_generic_rule(rule, records, result)
            
            # Calculate failure percentage and score
            if result["total_count"] > 0:
                result["failure_percentage"] = result["failure_count"] / result["total_count"]
                result["score"] = 1.0 - result["failure_percentage"]
            
            # Determine if this constitutes an issue
            failure_pct = result["failure_percentage"]
            result["has_issues"] = failure_pct > rule.error_threshold
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.id}: {e}")
            result["score"] = 0.0
            result["has_issues"] = True
            result["details"]["error"] = str(e)
        
        return result
    
    async def _evaluate_completeness_rule(self, rule: QualityRule, records: List[Dict],
                                        result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate completeness rule"""
        required_fields = rule.parameters.get('required_fields', [])
        failures = []
        
        for i, record in enumerate(records):
            for field in required_fields:
                value = record.get(field)
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    failures.append({"record_index": i, "field": field, "value": value})
                    result["failure_count"] += 1
                    
                    if len(result["sample_failures"]) < self.max_sample_failures:
                        result["sample_failures"].append({
                            "field": field,
                            "record_index": i,
                            "issue": "missing_value"
                        })
        
        result["details"]["required_fields"] = required_fields
        result["details"]["total_field_checks"] = len(required_fields) * len(records)
        
        return result
    
    async def _evaluate_regex_rule(self, rule: QualityRule, records: List[Dict],
                                 result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate regex validation rule"""
        pattern = rule.parameters.get('pattern', '')
        if not pattern:
            return result
        
        compiled_pattern = re.compile(pattern)
        target_columns = rule.columns or set()
        
        for i, record in enumerate(records):
            for column, value in record.items():
                if target_columns and column.lower() not in [c.lower() for c in target_columns]:
                    continue
                
                if value and isinstance(value, str):
                    if not compiled_pattern.match(value):
                        result["failure_count"] += 1
                        
                        if len(result["sample_failures"]) < self.max_sample_failures:
                            result["sample_failures"].append({
                                "column": column,
                                "value": value,
                                "record_index": i,
                                "issue": "invalid_format"
                            })
        
        # Adjust total count for targeted columns
        if target_columns:
            total_values = 0
            for record in records:
                for column in record:
                    if column.lower() in [c.lower() for c in target_columns]:
                        if record[column] is not None:
                            total_values += 1
            result["total_count"] = total_values
        
        return result
    
    async def _evaluate_uniqueness_rule(self, rule: QualityRule, records: List[Dict],
                                      result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate uniqueness rule"""
        key_columns = rule.parameters.get('key_columns', ['id'])
        seen_keys = set()
        
        for i, record in enumerate(records):
            # Create composite key
            key_parts = []
            for column in key_columns:
                value = record.get(column, '')
                key_parts.append(str(value))
            
            composite_key = '|'.join(key_parts)
            
            if composite_key in seen_keys:
                result["failure_count"] += 1
                
                if len(result["sample_failures"]) < self.max_sample_failures:
                    result["sample_failures"].append({
                        "key_columns": key_columns,
                        "key_value": composite_key,
                        "record_index": i,
                        "issue": "duplicate_key"
                    })
            else:
                seen_keys.add(composite_key)
        
        result["details"]["key_columns"] = key_columns
        result["details"]["unique_keys_found"] = len(seen_keys)
        
        return result
    
    async def _evaluate_numeric_range_rule(self, rule: QualityRule, records: List[Dict],
                                         result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate numeric range rule"""
        range_specs = rule.parameters
        
        for i, record in enumerate(records):
            for column, value in record.items():
                if column not in range_specs:
                    continue
                
                try:
                    numeric_value = float(value) if value is not None else None
                    if numeric_value is None:
                        continue
                    
                    range_spec = range_specs[column]
                    min_val = range_spec.get('min')
                    max_val = range_spec.get('max')
                    
                    out_of_range = False
                    if min_val is not None and numeric_value < min_val:
                        out_of_range = True
                    if max_val is not None and numeric_value > max_val:
                        out_of_range = True
                    
                    if out_of_range:
                        result["failure_count"] += 1
                        
                        if len(result["sample_failures"]) < self.max_sample_failures:
                            result["sample_failures"].append({
                                "column": column,
                                "value": numeric_value,
                                "range": range_spec,
                                "record_index": i,
                                "issue": "out_of_range"
                            })
                
                except (ValueError, TypeError):
                    # Non-numeric value in numeric column
                    result["failure_count"] += 1
                    
                    if len(result["sample_failures"]) < self.max_sample_failures:
                        result["sample_failures"].append({
                            "column": column,
                            "value": value,
                            "record_index": i,
                            "issue": "non_numeric_value"
                        })
        
        return result
    
    async def _evaluate_date_consistency_rule(self, rule: QualityRule, records: List[Dict],
                                            result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate date consistency rule"""
        date_checks = rule.parameters.get('date_order_checks', [])
        
        for i, record in enumerate(records):
            for earlier_field, later_field in date_checks:
                try:
                    earlier_value = record.get(earlier_field)
                    later_value = record.get(later_field)
                    
                    if not earlier_value or not later_value:
                        continue
                    
                    # Parse dates (simplified - would handle various formats in production)
                    if isinstance(earlier_value, str):
                        earlier_date = datetime.fromisoformat(earlier_value.replace('Z', '+00:00'))
                    else:
                        earlier_date = earlier_value
                    
                    if isinstance(later_value, str):
                        later_date = datetime.fromisoformat(later_value.replace('Z', '+00:00'))
                    else:
                        later_date = later_value
                    
                    if earlier_date > later_date:
                        result["failure_count"] += 1
                        
                        if len(result["sample_failures"]) < self.max_sample_failures:
                            result["sample_failures"].append({
                                "earlier_field": earlier_field,
                                "later_field": later_field,
                                "earlier_value": earlier_value,
                                "later_value": later_value,
                                "record_index": i,
                                "issue": "date_order_violation"
                            })
                
                except (ValueError, TypeError) as e:
                    # Date parsing error
                    result["failure_count"] += 1
                    
                    if len(result["sample_failures"]) < self.max_sample_failures:
                        result["sample_failures"].append({
                            "earlier_field": earlier_field,
                            "later_field": later_field,
                            "record_index": i,
                            "issue": "date_parsing_error",
                            "error": str(e)
                        })
        
        return result
    
    async def _evaluate_freshness_rule(self, rule: QualityRule, records: List[Dict],
                                     metadata: Optional[Dict[str, Any]],
                                     result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate data freshness rule"""
        max_age_hours = rule.parameters.get('max_age_hours', 24)
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Check metadata first for last update time
        if metadata:
            last_updated = metadata.get('last_updated') or metadata.get('updated_at')
            if last_updated:
                try:
                    if isinstance(last_updated, str):
                        update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    else:
                        update_time = last_updated
                    
                    if update_time < cutoff_time:
                        result["failure_count"] = len(records)  # All records are stale
                        result["sample_failures"].append({
                            "issue": "stale_data",
                            "last_updated": last_updated,
                            "max_age_hours": max_age_hours,
                            "age_hours": (datetime.utcnow() - update_time).total_seconds() / 3600
                        })
                
                except (ValueError, TypeError):
                    pass
        
        # Check individual record timestamps if available
        timestamp_fields = ['updated_at', 'created_at', 'timestamp', 'last_modified']
        
        for i, record in enumerate(records):
            record_timestamp = None
            
            for field in timestamp_fields:
                if field in record and record[field]:
                    try:
                        if isinstance(record[field], str):
                            record_timestamp = datetime.fromisoformat(record[field].replace('Z', '+00:00'))
                        else:
                            record_timestamp = record[field]
                        break
                    except (ValueError, TypeError):
                        continue
            
            if record_timestamp and record_timestamp < cutoff_time:
                result["failure_count"] += 1
                
                if len(result["sample_failures"]) < self.max_sample_failures:
                    age_hours = (datetime.utcnow() - record_timestamp).total_seconds() / 3600
                    result["sample_failures"].append({
                        "record_index": i,
                        "timestamp": record_timestamp.isoformat(),
                        "age_hours": age_hours,
                        "max_age_hours": max_age_hours,
                        "issue": "stale_record"
                    })
        
        return result
    
    async def _evaluate_generic_rule(self, rule: QualityRule, records: List[Dict],
                                   result: Dict[str, Any]) -> Dict[str, Any]:
        """Generic rule evaluation for custom rules"""
        # This would implement custom rule logic based on rule parameters
        # For now, just return a passing result
        result["details"]["rule_type"] = rule.rule_type
        result["details"]["message"] = "Generic rule evaluation - manual review recommended"
        return result
    
    def _create_quality_issue(self, rule: QualityRule, data_id: str,
                            rule_result: Dict[str, Any]) -> QualityIssue:
        """Create a quality issue from rule evaluation result"""
        
        # Map rule types to issue types
        issue_type_mapping = {
            "completeness": QualityIssueType.MISSING_VALUES,
            "regex": QualityIssueType.INVALID_FORMAT,
            "uniqueness": QualityIssueType.DUPLICATE_RECORDS,
            "numeric_range": QualityIssueType.OUTLIER_VALUES,
            "date_consistency": QualityIssueType.INCONSISTENT_VALUES,
            "freshness": QualityIssueType.STALE_DATA
        }
        
        issue_type = issue_type_mapping.get(rule.rule_type, QualityIssueType.SCHEMA_VIOLATION)
        
        # Determine severity based on failure percentage
        failure_pct = rule_result["failure_percentage"]
        if failure_pct > 0.5:
            severity = "critical"
        elif failure_pct > rule.error_threshold * 2:
            severity = "high"
        else:
            severity = rule.severity
        
        issue = QualityIssue(
            id=f"{data_id}_{rule.id}_{int(datetime.utcnow().timestamp())}",
            rule_id=rule.id,
            data_id=data_id,
            issue_type=issue_type,
            dimension=rule.dimension,
            description=f"{rule.name}: {failure_pct:.1%} failure rate",
            severity=severity,
            failure_count=rule_result["failure_count"],
            total_records=rule_result["total_count"],
            failure_percentage=failure_pct,
            sample_failures=rule_result["sample_failures"],
            metadata={
                "rule_name": rule.name,
                "rule_details": rule_result.get("details", {}),
                "remediation_suggestion": rule.remediation_suggestion
            }
        )
        
        return issue
    
    def _calculate_null_percentage(self, records: List[Dict]) -> float:
        """Calculate percentage of null/empty values"""
        if not records:
            return 0.0
        
        total_values = 0
        null_values = 0
        
        for record in records:
            for value in record.values():
                total_values += 1
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    null_values += 1
        
        return (null_values / total_values) * 100 if total_values > 0 else 0.0
    
    def _calculate_duplicate_percentage(self, records: List[Dict]) -> float:
        """Calculate percentage of duplicate records"""
        if not records:
            return 0.0
        
        # Create hashes of records for comparison
        record_hashes = []
        for record in records:
            record_str = json.dumps(record, sort_keys=True)
            record_hash = hashlib.md5(record_str.encode()).hexdigest()
            record_hashes.append(record_hash)
        
        unique_count = len(set(record_hashes))
        total_count = len(record_hashes)
        
        duplicate_count = total_count - unique_count
        return (duplicate_count / total_count) * 100 if total_count > 0 else 0.0
    
    def _generate_quality_recommendations(self, assessment: QualityAssessment) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Overall score recommendations
        if assessment.overall_score < self.critical_quality_threshold:
            recommendations.append("Critical quality issues detected - immediate attention required")
        elif assessment.overall_score < self.default_quality_threshold:
            recommendations.append("Quality score below threshold - review and improve data processes")
        
        # Issue-specific recommendations
        if assessment.critical_issues > 0:
            recommendations.append("Address critical quality issues first")
        
        if assessment.null_percentage > 20:
            recommendations.append("High null percentage detected - review data completeness")
        
        if assessment.duplicate_percentage > 5:
            recommendations.append("Significant duplicates found - implement deduplication process")
        
        # Dimension-specific recommendations
        for dimension, score in assessment.dimension_scores.items():
            if score < 0.7:
                recommendations.append(f"Improve {dimension.value} through targeted quality measures")
        
        return recommendations
    
    async def run_quality_checks(self):
        """Run periodic quality checks across monitored data"""
        try:
            self.logger.info("Running periodic data quality checks")
            
            # This would typically iterate through registered datasets
            # For now, log the activity
            check_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_rules": len(self.rules),
                "active_rules": len([r for r in self.rules.values() if r.is_active]),
                "total_issues": len(self.issues),
                "unresolved_issues": len([i for i in self.issues.values() if not i.resolved_at])
            }
            
            self.logger.info(f"Quality checks completed: {check_results}")
            
        except Exception as e:
            self.logger.error(f"Error in periodic quality checks: {e}")
    
    def resolve_issue(self, issue_id: str, resolution_method: str) -> bool:
        """Mark a quality issue as resolved"""
        issue = self.issues.get(issue_id)
        if issue:
            issue.resolved_at = datetime.utcnow()
            issue.resolution_method = resolution_method
            self.stats["issues_resolved"] += 1
            self.logger.info(f"Resolved quality issue {issue_id}: {resolution_method}")
            return True
        return False
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics"""
        stats = {
            "overall_statistics": self.stats.copy(),
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.is_active]),
            "total_issues": len(self.issues),
            "issues_by_severity": {},
            "issues_by_dimension": {},
            "issues_by_type": {},
            "recent_issues": [],
            "rule_performance": {},
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Analyze issues
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        
        for issue in self.issues.values():
            # Count by severity
            severity = issue.severity
            stats["issues_by_severity"][severity] = stats["issues_by_severity"].get(severity, 0) + 1
            
            # Count by dimension
            dimension = issue.dimension.value
            stats["issues_by_dimension"][dimension] = stats["issues_by_dimension"].get(dimension, 0) + 1
            
            # Count by type
            issue_type = issue.issue_type.value
            stats["issues_by_type"][issue_type] = stats["issues_by_type"].get(issue_type, 0) + 1
            
            # Recent issues
            if issue.detected_at >= recent_cutoff:
                stats["recent_issues"].append({
                    "id": issue.id,
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "description": issue.description,
                    "detected_at": issue.detected_at.isoformat(),
                    "resolved": issue.resolved_at is not None
                })
        
        # Rule performance
        for rule in self.rules.values():
            rule_issues = [i for i in self.issues.values() if i.rule_id == rule.id]
            stats["rule_performance"][rule.id] = {
                "name": rule.name,
                "dimension": rule.dimension.value,
                "issues_detected": len(rule_issues),
                "is_active": rule.is_active
            }
        
        # Sort recent issues by date
        stats["recent_issues"].sort(key=lambda x: x["detected_at"], reverse=True)
        stats["recent_issues"] = stats["recent_issues"][:20]  # Top 20
        
        return stats
    
    async def shutdown(self):
        """Shutdown the quality monitor"""
        try:
            self.logger.info("Shutting down data quality monitor")
            # Any cleanup tasks would go here
            self.logger.info("Data quality monitor shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during quality monitor shutdown: {e}")