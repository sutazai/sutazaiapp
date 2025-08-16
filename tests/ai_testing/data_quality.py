"""
Data Quality Testing Framework
Enterprise-grade data quality validation for AI systems
"""

import pytest
import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataQualityThresholds:
    """Data quality validation thresholds"""
    
    def __init__(self):
        self.max_missing_percentage = 0.05  # 5% maximum missing values
        self.min_unique_values = 2  # Minimum unique values per column
        self.max_duplicate_percentage = 0.01  # 1% maximum duplicates
        self.max_outlier_percentage = 0.05  # 5% maximum outliers

class Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestDataset:
    """Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test dataset generator for testing framework"""
    
    def __init__(self, rows: int = 1000, missing_rate: float = 0.02):
        np.random.seed(42)
        
        # Generate realistic data with controlled quality issues
        self.data = pd.DataFrame({
            'user_id': range(rows),
            'age': np.random.normal(35, 15, rows),
            'income': np.random.exponential(50000, rows),
            'score': np.random.beta(2, 5, rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'timestamp': pd.date_range('2024-01-01', periods=rows, freq='1H'),
            'email': [f'user{i}@example.com' for i in range(rows)],
            'phone': [f'+1-555-{1000+i:04d}' for i in range(rows)]
        })
        
        # Introduce controlled missing values
        missing_indices = np.random.choice(rows, int(rows * missing_rate), replace=False)
        self.data.loc[missing_indices, 'age'] = np.nan
        
        # Add some duplicates
        duplicate_indices = np.random.choice(rows//2, int(rows * 0.005), replace=False)
        for idx in duplicate_indices:
            if idx + 1 < len(self.data):
                self.data.iloc[idx + 1] = self.data.iloc[idx]

class DataQualityTestSuite:
    """Enterprise data quality testing framework"""
    
    def __init__(self, dataset: Optional[pd.DataFrame] = None, reference_dataset: Optional[pd.DataFrame] = None):
        if dataset is None:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_data = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestDataset()
            self.dataset = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_data.data
        else:
            self.dataset = dataset
            
        self.reference_dataset = reference_dataset
        self.thresholds = DataQualityThresholds()
        
    @pytest.mark.data_quality
    @pytest.mark.unit
    def test_data_completeness_validation(self):
        """Test data completeness - Zero tolerance for excessive missing data"""
        logger.info("Running data completeness validation")
        
        missing_analysis = {}
        total_cells = len(self.dataset) * len(self.dataset.columns)
        
        for column in self.dataset.columns:
            missing_count = self.dataset[column].isnull().sum()
            missing_percentage = missing_count / len(self.dataset)
            missing_analysis[column] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage
            }
            
            # Completeness assertion per column
            assert missing_percentage <= self.thresholds.max_missing_percentage, \
                f"Column '{column}' has {missing_percentage*100:.2f}% missing values, exceeds {self.thresholds.max_missing_percentage*100}% threshold"
        
        # Overall completeness check
        total_missing = self.dataset.isnull().sum().sum()
        overall_completeness = 1 - (total_missing / total_cells)
        
        assert overall_completeness >= 0.95, \
            f"Overall data completeness {overall_completeness*100:.2f}% below 95% threshold"
            
        logger.info(f"✅ Data completeness validation passed: {overall_completeness*100:.2f}% complete")
        
    @pytest.mark.data_quality
    @pytest.mark.unit
    def test_data_uniqueness_validation(self):
        """Test data uniqueness and duplicate detection"""
        logger.info("Running data uniqueness validation")
        
        # Check for duplicate rows
        duplicate_count = self.dataset.duplicated().sum()
        duplicate_percentage = duplicate_count / len(self.dataset)
        
        assert duplicate_percentage <= self.thresholds.max_duplicate_percentage, \
            f"Duplicate rows {duplicate_percentage*100:.2f}% exceeds {self.thresholds.max_duplicate_percentage*100}% threshold"
        
        # Check unique identifier columns
        if 'user_id' in self.dataset.columns:
            unique_ids = self.dataset['user_id'].nunique()
            expected_unique = len(self.dataset) - duplicate_count
            
            assert unique_ids >= expected_unique * 0.99, \
                f"User ID uniqueness compromised: {unique_ids} unique vs {expected_unique} expected"
        
        logger.info(f"✅ Data uniqueness validation passed: {duplicate_percentage*100:.2f}% duplicates")
        
    @pytest.mark.data_quality
    @pytest.mark.unit
    def test_data_consistency_validation(self):
        """Test data consistency and format validation"""
        logger.info("Running data consistency validation")
        
        consistency_checks = []
        
        # Email format validation
        if 'email' in self.dataset.columns:
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            valid_emails = self.dataset['email'].dropna().apply(lambda x: bool(email_pattern.match(str(x))))
            email_validity = valid_emails.mean()
            
            assert email_validity >= 0.95, f"Email format consistency below 95%: {email_validity*100:.2f}%"
            consistency_checks.append(f"Email format: {email_validity*100:.2f}%")
        
        # Phone format validation
        if 'phone' in self.dataset.columns:
            phone_pattern = re.compile(r'^\+1-\d{3}-\d{4}$')
            valid_phones = self.dataset['phone'].dropna().apply(lambda x: bool(phone_pattern.match(str(x))))
            phone_validity = valid_phones.mean()
            
            assert phone_validity >= 0.95, f"Phone format consistency below 95%: {phone_validity*100:.2f}%"
            consistency_checks.append(f"Phone format: {phone_validity*100:.2f}%")
        
        # Numeric range validation
        if 'age' in self.dataset.columns:
            valid_ages = self.dataset['age'].dropna()
            age_validity = ((valid_ages >= 0) & (valid_ages <= 120)).mean()
            
            assert age_validity >= 0.98, f"Age range consistency below 98%: {age_validity*100:.2f}%"
            consistency_checks.append(f"Age range: {age_validity*100:.2f}%")
        
        logger.info(f"✅ Data consistency validation passed: {', '.join(consistency_checks)}")
        
    @pytest.mark.data_quality
    @pytest.mark.unit
    def test_data_distribution_validation(self):
        """Test data distribution and outlier detection"""
        logger.info("Running data distribution validation")
        
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in self.dataset.columns and not self.dataset[column].empty:
                column_data = self.dataset[column].dropna()
                
                if len(column_data) > 0:
                    # Calculate z-scores for outlier detection
                    z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                    outliers = z_scores > 3  # 3 standard deviations
                    outlier_percentage = outliers.mean()
                    
                    assert outlier_percentage <= self.thresholds.max_outlier_percentage, \
                        f"Column '{column}' has {outlier_percentage*100:.2f}% outliers, exceeds {self.thresholds.max_outlier_percentage*100}% threshold"
        
        logger.info("✅ Data distribution validation passed")
        
    @pytest.mark.data_quality
    @pytest.mark.security
    def test_data_privacy_validation(self):
        """Test data privacy compliance - Security by Design"""
        logger.info("Running data privacy validation")
        
        # PII detection patterns
        pii_patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'sensitive_text': re.compile(r'\b(password|secret|token|key)\b', re.IGNORECASE)
        }
        
        pii_violations = []
        
        # Check text columns for PII patterns
        text_columns = self.dataset.select_dtypes(include=['object', 'string']).columns
        
        for column in text_columns:
            column_data = self.dataset[column].dropna().astype(str)
            
            for pii_type, pattern in pii_patterns.items():
                matches = column_data.str.contains(pattern, regex=True).sum()
                if matches > 0:
                    pii_violations.append(f"Column '{column}' contains {matches} {pii_type} patterns")
        
        # Assert no PII violations
        assert len(pii_violations) == 0, f"PII detected in dataset: {'; '.join(pii_violations)}"
        
        logger.info("✅ Data privacy validation passed - No PII detected")
        
    @pytest.mark.data_quality
    @pytest.mark.integration
    def test_data_drift_detection(self):
        """Test for data drift against reference dataset"""
        logger.info("Running data drift detection")
        
        if self.reference_dataset is None:
            logger.warning("No reference dataset provided, skipping drift detection")
            pytest.skip("No reference dataset for drift detection")
            
        drift_results = {}
        
        # Compare distributions for numeric columns
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
        common_columns = set(numeric_columns) & set(self.reference_dataset.columns)
        
        for column in common_columns:
            current_data = self.dataset[column].dropna()
            reference_data = self.reference_dataset[column].dropna()
            
            if len(current_data) > 0 and len(reference_data) > 0:
                # Simple drift detection using mean and std comparison
                current_mean = current_data.mean()
                reference_mean = reference_data.mean()
                
                current_std = current_data.std()
                reference_std = reference_data.std()
                
                # Calculate drift scores
                mean_drift = abs(current_mean - reference_mean) / reference_mean if reference_mean != 0 else 0
                std_drift = abs(current_std - reference_std) / reference_std if reference_std != 0 else 0
                
                drift_results[column] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'significant_drift': mean_drift > 0.1 or std_drift > 0.2
                }
        
        # Check for significant drift
        significant_drift_columns = [col for col, result in drift_results.items() if result['significant_drift']]
        
        assert len(significant_drift_columns) == 0, \
            f"Significant data drift detected in columns: {significant_drift_columns}"
        
        logger.info(f"✅ Data drift detection passed - {len(drift_results)} columns analyzed")
        
    @pytest.mark.data_quality
    @pytest.mark.performance
    def test_data_processing_performance(self):
        """Test data processing performance"""
        logger.info("Running data processing performance test")
        
        import time
        
        # Test data loading performance
        start_time = time.time()
        dataset_copy = self.dataset.copy()
        load_time = time.time() - start_time
        
        assert load_time < 1.0, f"Data loading too slow: {load_time:.3f}s"
        
        # Test basic operations performance
        start_time = time.time()
        summary_stats = dataset_copy.describe()
        stats_time = time.time() - start_time
        
        assert stats_time < 0.5, f"Statistics calculation too slow: {stats_time:.3f}s"
        
        # Test data filtering performance
        start_time = time.time()
        filtered_data = dataset_copy[dataset_copy.columns[0]].dropna()
        filter_time = time.time() - start_time
        
        assert filter_time < 0.1, f"Data filtering too slow: {filter_time:.3f}s"
        
        logger.info(f"✅ Data processing performance test passed: Load {load_time:.3f}s, Stats {stats_time:.3f}s, Filter {filter_time:.3f}s")

# Pytest fixtures for data quality testing
@pytest.fixture
def sample_dataset():
    """Fixture providing sample dataset for testing"""
    Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_data = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestDataset(rows=500, missing_rate=0.01)
    return Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_data.data

@pytest.fixture
def reference_dataset():
    """Fixture providing reference dataset for drift detection"""
    Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_data = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real TestDataset(rows=500, missing_rate=0.01)
    return Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_data.data

@pytest.fixture
def data_quality_suite(sample_dataset, reference_dataset):
    """Fixture providing complete data quality test suite"""
    return DataQualityTestSuite(sample_dataset, reference_dataset)

# Test class using fixtures
class TestDataQualityValidation:
    """Test class for data quality validation using pytest fixtures"""
    
    def test_completeness_with_fixture(self, data_quality_suite):
        """Test data completeness using pytest fixture"""
        data_quality_suite.test_data_completeness_validation()
        
    def test_uniqueness_with_fixture(self, data_quality_suite):
        """Test data uniqueness using pytest fixture"""
        data_quality_suite.test_data_uniqueness_validation()
        
    def test_consistency_with_fixture(self, data_quality_suite):
        """Test data consistency using pytest fixture"""
        data_quality_suite.test_data_consistency_validation()
        
    def test_distribution_with_fixture(self, data_quality_suite):
        """Test data distribution using pytest fixture"""
        data_quality_suite.test_data_distribution_validation()
        
    def test_privacy_with_fixture(self, data_quality_suite):
        """Test data privacy using pytest fixture"""
        data_quality_suite.test_data_privacy_validation()
        
    def test_drift_with_fixture(self, data_quality_suite):
        """Test data drift using pytest fixture"""
        data_quality_suite.test_data_drift_detection()
        
    def test_performance_with_fixture(self, data_quality_suite):
        """Test data processing performance using pytest fixture"""
        data_quality_suite.test_data_processing_performance()