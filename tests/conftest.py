"""
Pytest configuration and shared fixtures for heart failure prediction testing.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

@pytest.fixture
def sample_data():
    """Create sample heart disease data for testing."""
    data = {
        'Age': [45, 67, 29, 56, 78, 34, 52, 43, 61, 38],
        'Sex': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        'ChestPainType': [2, 1, 3, 0, 2, 1, 3, 0, 2, 1],
        'RestingBP': [130, 145, 120, 140, 160, 115, 135, 125, 150, 122],
        'Cholesterol': [250, 200, 180, 280, 320, 150, 240, 190, 300, 170],
        'FastingBS': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        'RestingECG': [0, 1, 0, 2, 1, 0, 1, 0, 2, 0],
        'MaxHR': [150, 130, 180, 120, 100, 170, 140, 160, 110, 175],
        'ExerciseAngina': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        'Oldpeak': [1.2, 2.5, 0.0, 3.1, 4.0, 0.5, 1.8, 0.8, 2.9, 0.2],
        'ST_Slope': [1, 2, 0, 2, 2, 1, 1, 0, 2, 1],
        'HeartDisease': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_patient():
    """Create a sample patient data for testing predictions."""
    return {
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

@pytest.fixture
def test_database():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create test database with sample data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create hospitals table
    cursor.execute('''
        CREATE TABLE hospitals (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            zip_code TEXT,
            phone TEXT,
            website TEXT,
            cardiology_services INTEGER DEFAULT 1,
            emergency_services INTEGER DEFAULT 1,
            rating REAL DEFAULT 4.0,
            specialties TEXT
        )
    ''')
    
    # Insert sample hospital data
    hospitals_data = [
        (1, 'Test Heart Hospital', '123 Medical St', 'Test City', 'TS', '12345', 
         '555-0123', 'www.testheart.com', 1, 1, 4.5, 'Cardiology,Emergency'),
        (2, 'General Test Medical', '456 Health Ave', 'Test City', 'TS', '12346',
         '555-0124', 'www.testgeneral.com', 1, 1, 4.2, 'Cardiology,Internal Medicine')
    ]
    
    cursor.executemany('''
        INSERT INTO hospitals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    ''', hospitals_data)
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)

@pytest.fixture
def mock_model():
    """Create a mock machine learning model for testing."""
    model = Mock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    return model

@pytest.fixture
def test_data_dir():
    """Create temporary directory with test data files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create a test CSV file
    test_data = pd.DataFrame({
        'Age': [45, 67, 29],
        'Sex': [1, 0, 1],
        'ChestPainType': [2, 1, 3],
        'RestingBP': [130, 145, 120],
        'Cholesterol': [250, 200, 180],
        'FastingBS': [0, 1, 0],
        'RestingECG': [0, 1, 0],
        'MaxHR': [150, 130, 180],
        'ExerciseAngina': [0, 1, 0],
        'Oldpeak': [1.2, 2.5, 0.0],
        'ST_Slope': [1, 2, 0],
        'HeartDisease': [0, 1, 0]
    })
    
    test_csv_path = os.path.join(temp_dir, 'test_heart.csv')
    test_data.to_csv(test_csv_path, index=False)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_model_dir():
    """Create temporary directory for model testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    with patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.write') as mock_write:
        
        mock_sidebar.selectbox.return_value = 'Individual Prediction'
        mock_sidebar.button.return_value = False
        mock_columns.return_value = [Mock(), Mock()]
        
        yield {
            'sidebar': mock_sidebar,
            'columns': mock_columns,
            'success': mock_success,
            'error': mock_error,
            'write': mock_write
        }

# Test categories for marking
@pytest.fixture(scope="session")
def test_categories():
    """Define test categories for organized testing."""
    return {
        'unit': 'Unit tests for individual functions',
        'integration': 'Integration tests for component interaction',
        'model': 'Machine learning model tests',
        'database': 'Database operation tests',
        'streamlit': 'Streamlit app tests',
        'performance': 'Performance and load tests',
        'edge_cases': 'Edge case and error handling tests'
    }

# Custom markers for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "model: Model tests")
    config.addinivalue_line("markers", "database: Database tests") 
    config.addinivalue_line("markers", "streamlit: Streamlit tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "edge_cases: Edge case tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
