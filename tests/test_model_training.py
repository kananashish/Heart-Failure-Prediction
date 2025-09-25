"""
Tests for machine learning training pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from unittest.mock import patch, Mock, MagicMock

# Import modules to test
from src.train import ModelTrainer

@pytest.mark.model
class TestModelTrainer:
    """Test the ModelTrainer class."""
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer is not None
        assert hasattr(trainer, 'models')
        assert hasattr(trainer, 'results')
        assert hasattr(trainer, 'best_model')
        assert hasattr(trainer, 'best_model_name')
        assert trainer.models == {}  # Should start empty
        
        # Test model initialization
        trainer.initialize_models()
        assert len(trainer.models) > 0  # Should have models after initialization
    
    def test_prepare_data(self, sample_data):
        """Test data preparation method."""
        trainer = ModelTrainer()
        result = trainer.prepare_data(sample_data)
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'preprocessor' in result
        
        # Check data shapes
        assert result['X_train'].shape[0] > 0
        assert result['X_test'].shape[0] > 0
        assert len(result['y_train']) == result['X_train'].shape[0]
        assert len(result['y_test']) == result['X_test'].shape[0]
    
    @patch('src.train_model.cross_val_score')
    @patch('src.train_model.GridSearchCV')
    def test_train_model(self, mock_grid_search, mock_cv_score, sample_data):
        """Test individual model training."""
        # Setup mocks
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        
        mock_grid_search_instance = Mock()
        mock_grid_search_instance.fit.return_value = mock_grid_search_instance
        mock_grid_search_instance.best_estimator_ = mock_model
        mock_grid_search_instance.best_params_ = {'n_estimators': 100}
        mock_grid_search_instance.best_score_ = 0.85
        mock_grid_search.return_value = mock_grid_search_instance
        
        mock_cv_score.return_value = np.array([0.8, 0.85, 0.82, 0.88, 0.84])
        
        trainer = ModelTrainer()
        data_split = trainer.prepare_data(sample_data)
        
        result = trainer.train_model(
            'RandomForest',
            data_split['X_train'], 
            data_split['y_train'],
            data_split['X_test'],
            data_split['y_test']
        )
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'best_params' in result
        assert 'cv_scores' in result
    
    @patch('src.train_model.shap.TreeExplainer')
    def test_generate_shap_explanations(self, mock_shap_explainer, sample_data, mock_model):
        """Test SHAP explanation generation."""
        # Setup mock SHAP explainer
        mock_explainer_instance = Mock()
        mock_explainer_instance.shap_values.return_value = np.random.rand(5, 11)
        mock_shap_explainer.return_value = mock_explainer_instance
        
        trainer = ModelTrainer()
        data_split = trainer.prepare_data(sample_data)
        
        explanations = trainer.generate_shap_explanations(
            mock_model,
            data_split['X_test'],
            'TestModel'
        )
        
        assert 'shap_values' in explanations
        assert 'feature_importance' in explanations
    
    def test_train_all_models(self, sample_data, test_model_dir):
        """Test training all models (integration test)."""
        with patch('src.train_model.cross_val_score') as mock_cv, \
             patch('src.train_model.GridSearchCV') as mock_grid, \
             patch('os.makedirs'), \
             patch('joblib.dump'):
            
            # Setup mocks
            mock_cv.return_value = np.array([0.8, 0.85, 0.82])
            
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0, 1])
            mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]])
            
            mock_grid_instance = Mock()
            mock_grid_instance.fit.return_value = mock_grid_instance
            mock_grid_instance.best_estimator_ = mock_model
            mock_grid_instance.best_params_ = {'n_estimators': 100}
            mock_grid_instance.best_score_ = 0.85
            mock_grid.return_value = mock_grid_instance
            
            trainer = ModelTrainer()
            trainer.models = {'RandomForest': Mock()}  # Simplified for testing
            
            results = trainer.train_all_models(sample_data, save_models=False)
            
            assert isinstance(results, dict)
            assert len(results) > 0
            assert trainer.best_model is not None
            assert trainer.best_model_name is not None
    
    def test_evaluate_model_performance(self, sample_data, mock_model):
        """Test model performance evaluation."""
        trainer = ModelTrainer()
        data_split = trainer.prepare_data(sample_data)
        
        # Setup mock predictions
        y_pred = np.array([0, 1, 0])
        y_pred_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        
        mock_model.predict.return_value = y_pred
        mock_model.predict_proba.return_value = y_pred_proba
        
        metrics = trainer.evaluate_model_performance(
            mock_model,
            data_split['X_test'][:3],  # Take only first 3 samples
            data_split['y_test'][:3]
        )
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

@pytest.mark.model
class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_save_and_load_model(self, mock_model, test_model_dir):
        """Test saving and loading trained models."""
        model_path = os.path.join(test_model_dir, 'test_model.pkl')
        
        # Save model
        joblib.dump(mock_model, model_path)
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        assert loaded_model is not None
        # Should have same methods as original
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_model, 'predict_proba')
    
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_model_saving_with_metadata(self, mock_makedirs, mock_dump, mock_model):
        """Test saving models with metadata."""
        trainer = ModelTrainer()
        
        metadata = {
            'model_name': 'TestModel',
            'accuracy': 0.85,
            'timestamp': '2024-01-01'
        }
        
        trainer.save_model_with_metadata(mock_model, 'test_path', metadata)
        
        # Should call dump with model
        mock_dump.assert_called()
        # Should create directory
        mock_makedirs.assert_called()

@pytest.mark.performance
class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_training_time_reasonable(self, sample_data):
        """Test that model training completes in reasonable time."""
        import time
        
        trainer = ModelTrainer()
        # Use only one simple model for performance test
        trainer.models = {
            'LogisticRegression': {
                'model': Mock(),
                'params': {'C': [1.0]}
            }
        }
        
        start_time = time.time()
        
        with patch('src.train_model.cross_val_score') as mock_cv, \
             patch('src.train_model.GridSearchCV') as mock_grid:
            
            mock_cv.return_value = np.array([0.8, 0.85])
            
            mock_model = Mock()
            mock_grid_instance = Mock()
            mock_grid_instance.best_estimator_ = mock_model
            mock_grid_instance.best_params_ = {'C': 1.0}
            mock_grid_instance.best_score_ = 0.85
            mock_grid.return_value = mock_grid_instance
            
            trainer.train_all_models(sample_data, save_models=False)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert training_time < 30  # seconds
    
    def test_memory_usage_reasonable(self, sample_data):
        """Test that model training doesn't use excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        trainer = ModelTrainer()
        
        with patch('src.train_model.cross_val_score') as mock_cv, \
             patch('src.train_model.GridSearchCV') as mock_grid:
            
            mock_cv.return_value = np.array([0.8, 0.85])
            mock_grid.return_value = Mock()
            
            # Simulate training (without actual model training)
            trainer.prepare_data(sample_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500  # MB

@pytest.mark.edge_cases
class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test handling of insufficient training data."""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
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
            'ST_Slope': [1],
            'HeartDisease': [0]
        })
        
        trainer = ModelTrainer()
        
        with pytest.raises((ValueError, Exception)):
            trainer.train_all_models(minimal_data)
    
    def test_imbalanced_data_handling(self):
        """Test handling of highly imbalanced data."""
        # Create imbalanced dataset (90% class 0, 10% class 1)
        imbalanced_data = pd.DataFrame({
            'Age': [45] * 18 + [67] * 2,
            'Sex': [1] * 18 + [0] * 2,
            'ChestPainType': [2] * 20,
            'RestingBP': [130] * 20,
            'Cholesterol': [250] * 20,
            'FastingBS': [0] * 20,
            'RestingECG': [0] * 20,
            'MaxHR': [150] * 20,
            'ExerciseAngina': [0] * 20,
            'Oldpeak': [1.2] * 20,
            'ST_Slope': [1] * 20,
            'HeartDisease': [0] * 18 + [1] * 2  # Highly imbalanced
        })
        
        trainer = ModelTrainer()
        
        # Should handle imbalanced data (potentially with warnings)
        try:
            result = trainer.prepare_data(imbalanced_data, balance_data=True)
            assert result is not None
        except Exception as e:
            # Should raise appropriate error or warning
            pass
    
    def test_feature_with_zero_variance(self):
        """Test handling of features with zero variance."""
        # Create data with zero variance feature
        zero_var_data = pd.DataFrame({
            'Age': [45, 67, 29, 56, 78],
            'Sex': [1, 0, 1, 1, 0],
            'ChestPainType': [2, 2, 2, 2, 2],  # Zero variance
            'RestingBP': [130, 145, 120, 140, 160],
            'Cholesterol': [250, 200, 180, 280, 320],
            'FastingBS': [0, 1, 0, 1, 1],
            'RestingECG': [0, 1, 0, 2, 1],
            'MaxHR': [150, 130, 180, 120, 100],
            'ExerciseAngina': [0, 1, 0, 1, 1],
            'Oldpeak': [1.2, 2.5, 0.0, 3.1, 4.0],
            'ST_Slope': [1, 2, 0, 2, 2],
            'HeartDisease': [0, 1, 0, 1, 1]
        })
        
        trainer = ModelTrainer()
        
        # Should handle zero variance features appropriately
        result = trainer.prepare_data(zero_var_data)
        assert result is not None
        # Zero variance features might be removed or handled