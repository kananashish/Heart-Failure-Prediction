"""
Tests for data preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, Mock

# Import modules to test
from src.preprocess import HeartDiseasePreprocessor

@pytest.mark.unit
class TestHeartDiseasePreprocessor:
    """Test the HeartDiseasePreprocessor class."""
    
    def test_init(self):
        """Test HeartDiseasePreprocessor initialization."""
        preprocessor = HeartDiseasePreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'label_encoders')
        assert hasattr(preprocessor, 'scaler')
        assert hasattr(preprocessor, 'target_column')
        assert preprocessor.target_column == 'HeartDisease'
    
    def test_preprocess_with_valid_data(self, sample_data):
        """Test preprocessing with valid data."""
        preprocessor = HeartDiseasePreprocessor()
        
        # Use the preprocessor's fit_transform method
        X_processed, y_processed = preprocessor.fit_transform(sample_data)
        
        # Check output shapes
        assert X_processed.shape[0] == sample_data.shape[0]
        assert len(y_processed) == len(sample_data)
        
        # Check for no missing values
        assert not X_processed.isnull().any().any()
        assert not pd.isnull(y_processed).any()
    
    def test_preprocess_with_missing_values(self):
        """Test preprocessing handles missing values correctly."""
        # Create data with missing values
        data = pd.DataFrame({
            'Age': [45, np.nan, 67],
            'Sex': [1, 0, np.nan],
            'ChestPainType': [2, 1, 3],
            'RestingBP': [130, 145, 120],
            'Cholesterol': [250, np.nan, 180],
            'FastingBS': [0, 1, 0],
            'RestingECG': [0, 1, 0],
            'MaxHR': [150, 130, 180],
            'ExerciseAngina': [0, 1, 0],
            'Oldpeak': [1.2, 2.5, 0.0],
            'ST_Slope': [1, 2, 0]
        })
        y = pd.Series([0, 1, 0])
        
        preprocessor = Preprocessor()
        X_processed, y_processed = preprocessor.preprocess(data, y)
        
        # Should handle missing values (no NaN in output)
        assert not X_processed.isnull().any().any()
    
    def test_transform_single_sample(self, sample_data):
        """Test transforming a single sample."""
        preprocessor = Preprocessor()
        X = sample_data.drop('HeartDisease', axis=1)
        y = sample_data['HeartDisease']
        
        # Fit the preprocessor
        preprocessor.preprocess(X, y)
        
        # Transform single sample
        single_sample = X.iloc[0:1]
        transformed = preprocessor.transform(single_sample)
        
        assert transformed.shape[0] == 1
        assert not transformed.isnull().any().any()
    
    def test_column_alignment(self, sample_data):
        """Test that column alignment works correctly."""
        preprocessor = Preprocessor()
        X = sample_data.drop('HeartDisease', axis=1)
        
        # Test with missing columns
        X_missing = X.drop(['Age', 'Sex'], axis=1)
        
        # Should handle missing columns gracefully
        try:
            result = preprocessor.transform(X_missing)
            # If transform succeeds, result should have correct shape
            assert result is not None
        except Exception as e:
            # Or should raise appropriate error
            assert "column" in str(e).lower() or "feature" in str(e).lower()

@pytest.mark.unit  
class TestDataPrep:
    """Test data preparation functions."""
    
    def test_preprocess_data_basic(self, sample_data):
        """Test basic data preprocessing."""
        X = sample_data.drop('HeartDisease', axis=1)
        y = sample_data['HeartDisease']
        
        X_processed, y_processed = preprocess_data(X, y)
        
        assert X_processed.shape[0] == X.shape[0]
        assert len(y_processed) == len(y)
        assert not X_processed.isnull().any().any()
    
    def test_prepare_data_with_balancing(self, sample_data):
        """Test data preparation with balancing."""
        result = prepare_data(sample_data, balance_data=True)
        
        assert 'X_train' in result
        assert 'X_test' in result  
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'preprocessor' in result
        
        # Check shapes are reasonable
        assert result['X_train'].shape[1] == result['X_test'].shape[1]
        assert len(result['y_train']) == result['X_train'].shape[0]
        assert len(result['y_test']) == result['X_test'].shape[0]
    
    def test_prepare_data_without_balancing(self, sample_data):
        """Test data preparation without balancing."""
        result = prepare_data(sample_data, balance_data=False)
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result  
        assert 'y_test' in result
        assert 'preprocessor' in result
    
    @pytest.mark.parametrize("test_size", [0.2, 0.3, 0.4])
    def test_prepare_data_different_split_sizes(self, sample_data, test_size):
        """Test data preparation with different split sizes."""
        result = prepare_data(sample_data, test_size=test_size)
        
        total_samples = len(sample_data)
        expected_test_size = int(total_samples * test_size)
        
        # Allow some tolerance due to stratification
        assert abs(len(result['y_test']) - expected_test_size) <= 2

@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""
    
    def test_end_to_end_preprocessing(self, sample_data, test_data_dir):
        """Test complete preprocessing pipeline."""
        # Save sample data to temp file
        data_path = os.path.join(test_data_dir, 'integration_test.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Load and process data
        data = pd.read_csv(data_path)
        result = prepare_data(data, balance_data=True)
        
        # Verify complete pipeline works
        assert all(key in result for key in ['X_train', 'X_test', 'y_train', 'y_test', 'preprocessor'])
        
        # Test that preprocessor can transform new data
        X_new = data.drop('HeartDisease', axis=1).iloc[:2]
        X_transformed = result['preprocessor'].transform(X_new)
        
        assert X_transformed.shape[0] == 2
        assert X_transformed.shape[1] == result['X_train'].shape[1]

@pytest.mark.edge_cases
class TestPreprocessingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            preprocess_data(empty_df, pd.Series())
    
    def test_single_row_data(self):
        """Test handling of single row data."""
        single_row = pd.DataFrame({
            'Age': [45],
            'Sex': [1],
            'ChestPainType': [2],
            'RestingBP': [130],
            'Cholesterol': [250],
            'FastingBS': [0],
            'RestingECG': [0],
            'MaxHR': [150],
            'ExerciseAngina': [0],
            'Oldpeak': [1.2],
            'ST_Slope': [1]
        })
        y_single = pd.Series([0])
        
        # Should handle single row gracefully
        try:
            X_processed, y_processed = preprocess_data(single_row, y_single)
            assert X_processed.shape[0] == 1
            assert len(y_processed) == 1
        except ValueError:
            # Some preprocessing might not work with single samples
            pass
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        data = pd.DataFrame({
            'Age': [45, 67, 29],
            'Sex': [np.nan, np.nan, np.nan],  # All missing
            'ChestPainType': [2, 1, 3],
            'RestingBP': [130, 145, 120],
            'Cholesterol': [250, 200, 180],
            'FastingBS': [0, 1, 0],
            'RestingECG': [0, 1, 0],
            'MaxHR': [150, 130, 180],
            'ExerciseAngina': [0, 1, 0],
            'Oldpeak': [1.2, 2.5, 0.0],
            'ST_Slope': [1, 2, 0]
        })
        y = pd.Series([0, 1, 0])
        
        # Should handle all-missing column
        X_processed, y_processed = preprocess_data(data, y)
        
        # Should either impute or handle appropriately
        assert not X_processed.isnull().any().any()
    
    def test_inconsistent_data_types(self):
        """Test handling of inconsistent data types."""
        data = pd.DataFrame({
            'Age': ['45', '67', '29'],  # String instead of numeric
            'Sex': [1, 0, 1],
            'ChestPainType': [2, 1, 3],
            'RestingBP': [130, 145, 120],
            'Cholesterol': [250, 200, 180],
            'FastingBS': [0, 1, 0],
            'RestingECG': [0, 1, 0],
            'MaxHR': [150, 130, 180],
            'ExerciseAngina': [0, 1, 0],
            'Oldpeak': [1.2, 2.5, 0.0],
            'ST_Slope': [1, 2, 0]
        })
        y = pd.Series([0, 1, 0])
        
        # Should handle or raise appropriate error
        try:
            X_processed, y_processed = preprocess_data(data, y)
            # If successful, Age should be converted to numeric
            assert X_processed.dtypes['Age'] in [np.int64, np.float64]
        except (ValueError, TypeError):
            # Or should raise appropriate error
            pass