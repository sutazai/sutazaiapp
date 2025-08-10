"""
Data Classification System
=========================

Automatically classifies data based on sensitivity, type, and regulatory requirements.
Provides foundation for data lifecycle policies and access controls.
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Pattern
from dataclasses import dataclass, field
from datetime import datetime
import json


class DataClassification(Enum):
    """Data classification levels based on sensitivity and regulatory requirements"""
    PUBLIC = "public"              # No restrictions
    INTERNAL = "internal"          # Internal use only
    CONFIDENTIAL = "confidential"  # Restricted access required
    RESTRICTED = "restricted"      # Highest level of protection


class DataType(Enum):
    """Types of data handled by the system"""
    PII = "personally_identifiable_information"
    PHI = "protected_health_information"
    FINANCIAL = "financial_data"
    TECHNICAL = "technical_data"
    OPERATIONAL = "operational_data"
    AI_MODEL = "ai_model_data"
    COMMUNICATION = "communication_data"
    SYSTEM_LOG = "system_log"
    DOCUMENT = "document"
    METADATA = "metadata"


class RegulationScope(Enum):
    """Regulatory frameworks that apply to data"""
    GDPR = "gdpr"          # EU General Data Protection Regulation
    CCPA = "ccpa"          # California Consumer Privacy Act
    HIPAA = "hipaa"        # Health Insurance Portability and Accountability Act
    SOX = "sox"            # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"    # Payment Card Industry Data Security Standard
    NONE = "none"          # No specific regulation


@dataclass
class ClassificationRule:
    """Rule for classifying data based on patterns and metadata"""
    name: str
    patterns: List[Pattern[str]]
    data_types: Set[DataType]
    classification: DataClassification
    regulations: Set[RegulationScope]
    confidence: float = 1.0
    description: str = ""
    
    def __post_init__(self):
        """Compile regex patterns after initialization"""
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]


@dataclass
class ClassificationResult:
    """Result of data classification analysis"""
    classification: DataClassification
    data_types: Set[DataType]
    regulations: Set[RegulationScope]
    confidence: float
    matched_rules: List[str]
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    classified_at: datetime = field(default_factory=datetime.utcnow)


class DataClassifier:
    """
    Intelligent data classification system that automatically categorizes
    data based on content, metadata, and regulatory requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("data_classifier")
        self.classification_rules = self._initialize_classification_rules()
        
    def _initialize_classification_rules(self) -> List[ClassificationRule]:
        """Initialize predefined classification rules"""
        rules = []
        
        # PII Classification Rules
        rules.append(ClassificationRule(
            name="email_addresses",
            patterns=[r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            data_types={DataType.PII},
            classification=DataClassification.CONFIDENTIAL,
            regulations={RegulationScope.GDPR, RegulationScope.CCPA},
            description="Email addresses require confidential handling"
        ))
        
        rules.append(ClassificationRule(
            name="social_security_numbers",
            patterns=[r'\b\d{3}-?\d{2}-?\d{4}\b'],
            data_types={DataType.PII},
            classification=DataClassification.RESTRICTED,
            regulations={RegulationScope.GDPR, RegulationScope.CCPA},
            description="SSNs require highest level of protection"
        ))
        
        rules.append(ClassificationRule(
            name="phone_numbers", 
            patterns=[r'\b\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b'],
            data_types={DataType.PII},
            classification=DataClassification.CONFIDENTIAL,
            regulations={RegulationScope.GDPR, RegulationScope.CCPA},
            description="Phone numbers are confidential PII"
        ))
        
        # Financial Data Rules
        rules.append(ClassificationRule(
            name="credit_card_numbers",
            patterns=[r'\b(?:\d{4}[-\s]?){3}\d{4}\b'],
            data_types={DataType.FINANCIAL},
            classification=DataClassification.RESTRICTED,
            regulations={RegulationScope.PCI_DSS},
            description="Credit card numbers require PCI DSS compliance"
        ))
        
        # Health Information
        rules.append(ClassificationRule(
            name="medical_record_numbers",
            patterns=[r'\bMRN[:\s]*\d+\b', r'\bpatient[_\s]id[:\s]*\d+\b'],
            data_types={DataType.PHI},
            classification=DataClassification.RESTRICTED,
            regulations={RegulationScope.HIPAA},
            description="Medical records require HIPAA compliance"
        ))
        
        # AI Model Data
        rules.append(ClassificationRule(
            name="api_keys",
            patterns=[r'[a-zA-Z0-9]{32,}', r'sk-[a-zA-Z0-9]{48}'],
            data_types={DataType.TECHNICAL},
            classification=DataClassification.RESTRICTED,
            regulations={RegulationScope.NONE},
            description="API keys must be protected"
        ))
        
        # System Data
        rules.append(ClassificationRule(
            name="ip_addresses",
            patterns=[r'\b(?:\d{1,3}\.){3}\d{1,3}\b'],
            data_types={DataType.TECHNICAL, DataType.SYSTEM_LOG},
            classification=DataClassification.INTERNAL,
            regulations={RegulationScope.GDPR},
            description="IP addresses can be personal data under GDPR"
        ))
        
        # Agent Communication
        rules.append(ClassificationRule(
            name="agent_messages",
            patterns=[r'agent[_\s]message', r'llm[_\s]response', r'ai[_\s]generated'],
            data_types={DataType.AI_MODEL, DataType.COMMUNICATION},
            classification=DataClassification.INTERNAL,
            regulations={RegulationScope.NONE},
            description="AI agent communications are internal data"
        ))
        
        return rules
    
    def classify_data(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify data content and return classification result
        
        Args:
            data: Data content to classify
            metadata: Optional metadata about the data source
            
        Returns:
            ClassificationResult with classification details
        """
        if not data:
            return ClassificationResult(
                classification=DataClassification.PUBLIC,
                data_types=set(),
                regulations=set(),
                confidence=1.0,
                matched_rules=[],
                reasoning="Empty data classified as public"
            )
        
        matched_rules = []
        data_types = set()
        regulations = set()
        max_classification = DataClassification.PUBLIC
        total_confidence = 0.0
        
        # Apply classification rules
        for rule in self.classification_rules:
            matches = self._check_rule_match(data, rule)
            if matches:
                matched_rules.append(rule.name)
                data_types.update(rule.data_types)
                regulations.update(rule.regulations)
                total_confidence += rule.confidence
                
                # Use highest classification level found
                if self._classification_priority(rule.classification) > self._classification_priority(max_classification):
                    max_classification = rule.classification
        
        # Consider metadata-based classification
        if metadata:
            metadata_classification = self._classify_from_metadata(metadata)
            if self._classification_priority(metadata_classification) > self._classification_priority(max_classification):
                max_classification = metadata_classification
                matched_rules.append("metadata_based")
        
        # Calculate final confidence
        confidence = min(total_confidence / len(matched_rules), 1.0) if matched_rules else 0.5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(matched_rules, data_types, regulations)
        
        result = ClassificationResult(
            classification=max_classification,
            data_types=data_types,
            regulations=regulations,
            confidence=confidence,
            matched_rules=matched_rules,
            reasoning=reasoning,
            metadata=metadata or {}
        )
        
        self.logger.debug(f"Classified data as {max_classification.value} with confidence {confidence:.2f}")
        return result
    
    def _check_rule_match(self, data: str, rule: ClassificationRule) -> bool:
        """Check if data matches a classification rule"""
        for pattern in rule.compiled_patterns:
            if pattern.search(data):
                return True
        return False
    
    def _classification_priority(self, classification: DataClassification) -> int:
        """Return numeric priority for classification levels"""
        priority_map = {
            DataClassification.PUBLIC: 1,
            DataClassification.INTERNAL: 2,
            DataClassification.CONFIDENTIAL: 3,
            DataClassification.RESTRICTED: 4
        }
        return priority_map.get(classification, 1)
    
    def _classify_from_metadata(self, metadata: Dict[str, Any]) -> DataClassification:
        """Classify data based on metadata information"""
        source = metadata.get('source', '').lower()
        table_name = metadata.get('table_name', '').lower()
        schema = metadata.get('schema', '').lower()
        
        # Database table-based classification
        sensitive_tables = ['users', 'customers', 'payments', 'transactions', 'medical', 'health']
        if any(sensitive in table_name for sensitive in sensitive_tables):
            return DataClassification.CONFIDENTIAL
        
        # Source-based classification
        if 'external' in source or 'public' in source:
            return DataClassification.PUBLIC
        elif 'internal' in source:
            return DataClassification.INTERNAL
        elif any(keyword in source for keyword in ['secure', 'private', 'confidential']):
            return DataClassification.CONFIDENTIAL
        
        return DataClassification.INTERNAL
    
    def _generate_reasoning(self, matched_rules: List[str], data_types: Set[DataType], 
                          regulations: Set[RegulationScope]) -> str:
        """Generate human-readable reasoning for classification"""
        if not matched_rules:
            return "No sensitive patterns detected, classified as public data"
        
        reasoning_parts = []
        reasoning_parts.append(f"Matched {len(matched_rules)} classification rules: {', '.join(matched_rules)}")
        
        if data_types:
            type_names = [dt.value.replace('_', ' ').title() for dt in data_types]
            reasoning_parts.append(f"Detected data types: {', '.join(type_names)}")
        
        if regulations and RegulationScope.NONE not in regulations:
            reg_names = [reg.value.upper() for reg in regulations if reg != RegulationScope.NONE]
            reasoning_parts.append(f"Subject to regulations: {', '.join(reg_names)}")
        
        return ". ".join(reasoning_parts)
    
    def classify_database_schema(self, schema_info: Dict[str, Any]) -> Dict[str, ClassificationResult]:
        """
        Classify entire database schema including tables and columns
        
        Args:
            schema_info: Database schema information
            
        Returns:
            Dictionary mapping table.column to ClassificationResult
        """
        classifications = {}
        
        for table_name, table_info in schema_info.items():
            columns = table_info.get('columns', {})
            
            for column_name, column_info in columns.items():
                # Create mock data based on column name and type for classification
                mock_data = self._generate_mock_data_for_column(column_name, column_info)
                
                metadata = {
                    'source': 'database_schema',
                    'table_name': table_name,
                    'column_name': column_name,
                    'data_type': column_info.get('type', 'unknown'),
                    'schema': schema_info
                }
                
                result = self.classify_data(mock_data, metadata)
                classifications[f"{table_name}.{column_name}"] = result
        
        return classifications
    
    def _generate_mock_data_for_column(self, column_name: str, column_info: Dict[str, Any]) -> str:
        """Generate mock data for column classification based on name patterns"""
        column_name_lower = column_name.lower()
        
        # Generate mock data based on common column name patterns
        if 'email' in column_name_lower:
            return "user@example.com"
        elif 'phone' in column_name_lower:
            return "555-123-4567"
        elif 'ssn' in column_name_lower or 'social_security' in column_name_lower:
            return "123-45-6789"
        elif 'credit_card' in column_name_lower or 'cc_number' in column_name_lower:
            return "4111-1111-1111-1111"
        elif 'ip_address' in column_name_lower or column_name_lower == 'ip':
            return "192.168.1.1"
        elif 'api_key' in column_name_lower:
            return "sk-abcd1234567890abcd1234567890abcd1234567890abcd12"
        elif any(keyword in column_name_lower for keyword in ['password', 'token', 'secret']):
            return "sensitive_credential_data"
        else:
            return f"sample_{column_name_lower}_data"
    
    def get_retention_recommendations(self, classification: DataClassification,
                                    data_types: Set[DataType],
                                    regulations: Set[RegulationScope]) -> Dict[str, Any]:
        """
        Provide data retention recommendations based on classification
        
        Args:
            classification: Data classification level
            data_types: Types of data detected
            regulations: Applicable regulations
            
        Returns:
            Dictionary with retention recommendations
        """
        recommendations = {
            'min_retention_days': 0,
            'max_retention_days': None,
            'archive_after_days': None,
            'delete_after_days': None,
            'encryption_required': False,
            'backup_required': False,
            'audit_required': False,
            'reasoning': []
        }
        
        # Base recommendations by classification
        if classification == DataClassification.RESTRICTED:
            recommendations.update({
                'min_retention_days': 1095,  # 3 years
                'max_retention_days': 2555,  # 7 years
                'archive_after_days': 365,   # 1 year
                'encryption_required': True,
                'backup_required': True,
                'audit_required': True
            })
            recommendations['reasoning'].append("Restricted data requires extended retention and full security")
        
        elif classification == DataClassification.CONFIDENTIAL:
            recommendations.update({
                'min_retention_days': 365,   # 1 year
                'max_retention_days': 1095,  # 3 years
                'archive_after_days': 180,   # 6 months
                'encryption_required': True,
                'audit_required': True
            })
            recommendations['reasoning'].append("Confidential data needs secure handling and moderate retention")
        
        elif classification == DataClassification.INTERNAL:
            recommendations.update({
                'min_retention_days': 90,    # 3 months
                'max_retention_days': 365,   # 1 year
                'archive_after_days': 30,    # 1 month
                'backup_required': True
            })
            recommendations['reasoning'].append("Internal data follows standard business retention")
        
        else:  # PUBLIC
            recommendations.update({
                'min_retention_days': 0,
                'max_retention_days': 90,    # 3 months
                'delete_after_days': 90
            })
            recommendations['reasoning'].append("Public data can be retained briefly")
        
        # Adjust for specific regulations
        if RegulationScope.GDPR in regulations:
            recommendations['max_retention_days'] = min(recommendations.get('max_retention_days', 365), 1095)
            recommendations['delete_after_days'] = recommendations.get('max_retention_days')
            recommendations['reasoning'].append("GDPR requires data minimization and deletion rights")
        
        if RegulationScope.HIPAA in regulations:
            recommendations['min_retention_days'] = max(recommendations.get('min_retention_days', 0), 2190)  # 6 years
            recommendations['encryption_required'] = True
            recommendations['audit_required'] = True
            recommendations['reasoning'].append("HIPAA requires 6-year minimum retention and encryption")
        
        if RegulationScope.SOX in regulations:
            recommendations['min_retention_days'] = max(recommendations.get('min_retention_days', 0), 2555)  # 7 years
            recommendations['audit_required'] = True
            recommendations['reasoning'].append("SOX requires 7-year retention for financial records")
        
        return recommendations