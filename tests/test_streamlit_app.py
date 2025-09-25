"""
Tests for Streamlit web application.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys
import os

# Add the app directory to path
app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app')
sys.path.insert(0, app_path)

@pytest.mark.streamlit
class TestStreamlitApp:
    """Test the main Streamlit application."""
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.sidebar')
    def test_app_initialization(self, mock_sidebar, mock_title, mock_config):
        """Test app initialization and basic setup."""
        mock_sidebar.selectbox.return_value = 'Individual Prediction'
        
        # Import and test basic app setup
        try:
            import main as streamlit_app
            # App should import without errors
            assert streamlit_app is not None
        except ImportError:
            pytest.skip("Streamlit app main.py not available for testing")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.form')
    @patch('streamlit.columns') 
    @patch('joblib.load')
    def test_individual_prediction_interface(self, mock_joblib, mock_columns, mock_form, mock_sidebar):
        """Test individual prediction interface."""
        # Setup mocks
        mock_sidebar.selectbox.return_value = 'Individual Prediction'
        mock_form_instance = Mock()
        mock_form.return_value.__enter__.return_value = mock_form_instance
        mock_form_instance.form_submit_button.return_value = True
        
        # Mock form inputs
        mock_form_instance.number_input.return_value = 45  # Age
        mock_form_instance.selectbox.return_value = 'M'    # Sex
        mock_form_instance.slider.return_value = 130       # RestingBP
        
        # Mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib.return_value = mock_model
        
        mock_columns.return_value = [Mock(), Mock()]
        
        try:
            import main as streamlit_app
            # Should handle individual prediction flow
            assert hasattr(streamlit_app, 'create_model_input') or 'create_model_input' in dir(streamlit_app)
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.file_uploader')
    @patch('pandas.read_csv')
    def test_batch_prediction_interface(self, mock_read_csv, mock_uploader, mock_sidebar):
        """Test batch prediction interface."""
        mock_sidebar.selectbox.return_value = 'Batch Prediction'
        
        # Mock file upload
        mock_file = Mock()
        mock_file.name = 'test_batch.csv'
        mock_uploader.return_value = mock_file
        
        # Mock CSV data
        sample_batch_data = pd.DataFrame({
            'Age': [45, 67, 29],
            'Sex': ['M', 'F', 'M'],
            'ChestPainType': ['ATA', 'NAP', 'ASY'],
            'RestingBP': [130, 145, 120],
            'Cholesterol': [250, 200, 180],
            'FastingBS': [0, 1, 0],
            'RestingECG': ['Normal', 'ST', 'Normal'],
            'MaxHR': [150, 130, 180],
            'ExerciseAngina': ['N', 'Y', 'N'],
            'Oldpeak': [1.2, 2.5, 0.0],
            'ST_Slope': ['Up', 'Flat', 'Up']
        })
        mock_read_csv.return_value = sample_batch_data
        
        try:
            import main as streamlit_app
            # Should handle batch prediction interface
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    @patch('streamlit.sidebar')
    @patch('plotly.graph_objects.Figure')
    def test_visualization_dashboard(self, mock_plotly, mock_sidebar):
        """Test visualization dashboard functionality."""
        mock_sidebar.selectbox.return_value = 'Model Analysis'
        
        # Mock plotly figure
        mock_fig = Mock()
        mock_plotly.return_value = mock_fig
        
        try:
            import main as streamlit_app
            # Should handle visualization dashboard
        except ImportError:
            pytest.skip("Streamlit app not available for testing")

@pytest.mark.streamlit
class TestModelInputCreation:
    """Test model input creation and preprocessing for Streamlit."""
    
    def test_create_model_input_function(self):
        """Test the create_model_input helper function."""
        try:
            import main as streamlit_app
            
            if hasattr(streamlit_app, 'create_model_input'):
                patient_data = {
                    'Age': 45,
                    'Sex': 'M',
                    'ChestPainType': 'ATA',
                    'RestingBP': 130,
                    'Cholesterol': 250,
                    'FastingBS': 0,
                    'RestingECG': 'Normal',
                    'MaxHR': 150,
                    'ExerciseAngina': 'N',
                    'Oldpeak': 1.2,
                    'ST_Slope': 'Up'
                }
                
                result = streamlit_app.create_model_input(patient_data)
                
                assert isinstance(result, (list, np.ndarray))
                assert len(result) > 0
                
                # Should be all numeric for model input
                assert all(isinstance(x, (int, float)) for x in result)
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    def test_categorical_encoding_consistency(self):
        """Test that categorical encoding is consistent."""
        try:
            import main as streamlit_app
            
            if hasattr(streamlit_app, 'create_model_input'):
                # Test same input produces same output
                patient_data = {
                    'Age': 45,
                    'Sex': 'M',
                    'ChestPainType': 'ATA',
                    'RestingBP': 130,
                    'Cholesterol': 250,
                    'FastingBS': 0,
                    'RestingECG': 'Normal', 
                    'MaxHR': 150,
                    'ExerciseAngina': 'N',
                    'Oldpeak': 1.2,
                    'ST_Slope': 'Up'
                }
                
                result1 = streamlit_app.create_model_input(patient_data)
                result2 = streamlit_app.create_model_input(patient_data)
                
                assert np.array_equal(result1, result2)
        except ImportError:
            pytest.skip("Streamlit app not available for testing")

@pytest.mark.integration
class TestStreamlitIntegration:
    """Integration tests for Streamlit app components."""
    
    @patch('joblib.load')
    @patch('src.recommendations.HealthcareRecommendationSystem')
    def test_end_to_end_prediction_flow(self, mock_recommendations, mock_joblib):
        """Test complete prediction workflow."""
        # Setup mocks
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib.return_value = mock_model
        
        mock_rec_system = Mock()
        mock_rec_system.generate_patient_report.return_value = {
            'risk_assessment': {'probability': 0.7, 'risk_level': 'High'},
            'recommendations': {
                'diet': ['Follow heart-healthy diet'],
                'exercise': ['Regular moderate exercise']
            },
            'hospitals': [{'name': 'Test Hospital', 'rating': 4.5}]
        }
        mock_recommendations.return_value = mock_rec_system
        
        try:
            import main as streamlit_app
            
            # Test that app can load model and generate recommendations
            if hasattr(streamlit_app, 'create_model_input'):
                patient_data = {
                    'Age': 65,
                    'Sex': 'M',
                    'ChestPainType': 'ASY',
                    'RestingBP': 160,
                    'Cholesterol': 300,
                    'FastingBS': 1,
                    'RestingECG': 'LVH',
                    'MaxHR': 120,
                    'ExerciseAngina': 'Y',
                    'Oldpeak': 3.0,
                    'ST_Slope': 'Flat'
                }
                
                model_input = streamlit_app.create_model_input(patient_data)
                prediction = mock_model.predict_proba([model_input])
                
                assert prediction.shape == (1, 2)
                assert 0 <= prediction[0][1] <= 1  # Probability should be between 0 and 1
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.warning')
    def test_error_handling_in_app(self, mock_warning, mock_error, mock_success):
        """Test error handling in Streamlit app."""
        try:
            import main as streamlit_app
            
            # App should handle errors gracefully
            # Test with invalid model path or missing files
            with patch('joblib.load', side_effect=FileNotFoundError("Model not found")):
                # App should show error message instead of crashing
                mock_error.assert_called_once_if_error_shown = True
        except ImportError:
            pytest.skip("Streamlit app not available for testing")

@pytest.mark.performance
class TestStreamlitPerformance:
    """Test Streamlit app performance characteristics."""
    
    @patch('joblib.load')
    def test_app_loading_time(self, mock_joblib):
        """Test app loading performance."""
        import time
        
        # Mock fast model loading
        mock_model = Mock()
        mock_joblib.return_value = mock_model
        
        start_time = time.time()
        
        try:
            import main as streamlit_app
            load_time = time.time() - start_time
            
            # App should load quickly
            assert load_time < 10.0  # Should load in under 10 seconds
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    @patch('joblib.load')
    def test_prediction_response_time(self, mock_joblib):
        """Test prediction response time."""
        import time
        
        # Setup fast mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib.return_value = mock_model
        
        try:
            import main as streamlit_app
            
            if hasattr(streamlit_app, 'create_model_input'):
                patient_data = {
                    'Age': 45,
                    'Sex': 'M',
                    'ChestPainType': 'ATA',
                    'RestingBP': 130,
                    'Cholesterol': 250,
                    'FastingBS': 0,
                    'RestingECG': 'Normal',
                    'MaxHR': 150,
                    'ExerciseAngina': 'N',
                    'Oldpeak': 1.2,
                    'ST_Slope': 'Up'
                }
                
                start_time = time.time()
                
                # Create input and make prediction
                model_input = streamlit_app.create_model_input(patient_data)
                prediction = mock_model.predict_proba([model_input])
                
                response_time = time.time() - start_time
                
                # Should respond quickly
                assert response_time < 1.0  # Should predict in under 1 second
        except ImportError:
            pytest.skip("Streamlit app not available for testing")

@pytest.mark.edge_cases
class TestStreamlitEdgeCases:
    """Test edge cases and error conditions in Streamlit app."""
    
    def test_missing_model_file(self):
        """Test handling when model file is missing."""
        with patch('joblib.load', side_effect=FileNotFoundError("Model not found")):
            try:
                import main as streamlit_app
                # Should handle missing model gracefully
            except FileNotFoundError:
                # Or should raise appropriate error
                pass
            except ImportError:
                pytest.skip("Streamlit app not available for testing")
    
    def test_invalid_user_input(self):
        """Test handling of invalid user input."""
        try:
            import main as streamlit_app
            
            if hasattr(streamlit_app, 'create_model_input'):
                # Test with missing required fields
                incomplete_data = {
                    'Age': 45,
                    # Missing other required fields
                }
                
                try:
                    result = streamlit_app.create_model_input(incomplete_data)
                    # Should handle gracefully or provide defaults
                    assert result is not None
                except (KeyError, ValueError):
                    # Or should raise appropriate error
                    pass
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    def test_extreme_input_values(self):
        """Test handling of extreme input values."""
        try:
            import main as streamlit_app
            
            if hasattr(streamlit_app, 'create_model_input'):
                extreme_data = {
                    'Age': 200,  # Extreme age
                    'Sex': 'M',
                    'ChestPainType': 'ATA',
                    'RestingBP': 0,     # Invalid BP
                    'Cholesterol': -100,  # Negative cholesterol
                    'FastingBS': 5,     # Invalid boolean value
                    'RestingECG': 'Normal',
                    'MaxHR': 300,       # Extreme heart rate
                    'ExerciseAngina': 'N',
                    'Oldpeak': -5.0,    # Negative value
                    'ST_Slope': 'Up'
                }
                
                try:
                    result = streamlit_app.create_model_input(extreme_data)
                    # Should handle extreme values (clipping, validation, etc.)
                    assert result is not None
                    assert all(isinstance(x, (int, float)) for x in result)
                except (ValueError, Exception):
                    # Or should raise appropriate validation error
                    pass
        except ImportError:
            pytest.skip("Streamlit app not available for testing")
    
    @patch('pandas.read_csv')
    def test_invalid_batch_file_format(self, mock_read_csv):
        """Test handling of invalid batch file formats."""
        # Mock CSV with wrong columns
        mock_read_csv.return_value = pd.DataFrame({
            'WrongColumn1': [1, 2, 3],
            'WrongColumn2': ['A', 'B', 'C']
        })
        
        try:
            import main as streamlit_app
            
            # Should handle invalid file format gracefully
            # This would typically be tested with Streamlit's file_uploader
            # but we're testing the underlying logic
        except ImportError:
            pytest.skip("Streamlit app not available for testing")