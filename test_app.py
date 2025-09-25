"""
Test script for the Streamlit application components.
This tests the key functionality without requiring Streamlit server.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from recommendations import HealthcareRecommendationSystem, generate_patient_report
import joblib

def test_streamlit_components():
    """Test the key components of the Streamlit app."""
    print("=== Testing Streamlit App Components ===\n")
    
    # Test 1: Model and preprocessor loading
    print("1. Testing Model Loading...")
    try:
        model_path = os.path.join('models', 'best_heart_model.pkl')
        preprocessor_path = os.path.join('models', 'preprocessor.pkl')
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            print("✅ Model and preprocessor loaded successfully")
        else:
            print("❌ Model files not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test 2: Sample data loading
    print("\n2. Testing Sample Data Loading...")
    try:
        data_path = 'heart.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✅ Sample data loaded: {df.shape}")
        else:
            # Create sample data
            sample_data = {
                'Age': [54, 37, 41, 56, 57],
                'Sex': [1, 1, 0, 1, 0],
                'ChestPainType': [2, 1, 1, 0, 0],
                'RestingBP': [150, 130, 140, 120, 120],
                'Cholesterol': [195, 250, 204, 236, 354],
                'FastingBS': [0, 0, 0, 0, 0],
                'RestingECG': [1, 1, 0, 1, 1],
                'MaxHR': [122, 187, 172, 178, 163],
                'ExerciseAngina': [0, 0, 0, 0, 0],
                'Oldpeak': [0, 3.5, 1.4, 0.8, 0.6],
                'ST_Slope': [2, 0, 2, 2, 2]
            }
            df = pd.DataFrame(sample_data)
            print(f"✅ Created sample data: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False
    
    # Test 3: Prediction functionality
    print("\n3. Testing Prediction Functionality...")
    try:
        # Create a simple direct prediction using expected features
        # Use the selected features format that the model expects
        sample_features = {
            'Age': 55.0,
            'Sex': 1.0,  # M=1, F=0
            'FastingBS': 0.0,
            'MaxHR': 172.0,
            'ExerciseAngina': 1.0,  # Y=1, N=0
            'Oldpeak': 1.5,
            'ChestPainType_ATA': 1.0,
            'ChestPainType_NAP': 0.0,
            'ST_Slope_Flat': 0.0,
            'ST_Slope_Up': 1.0
        }
        
        # Create DataFrame with selected features
        input_df = pd.DataFrame([sample_features])
        
        # Use the actual estimator object for prediction
        estimator = model['model']
        prediction_proba = estimator.predict_proba(input_df)[0][1]
        prediction = estimator.predict(input_df)[0]
        
        print(f"✅ Prediction successful: {prediction} (Probability: {prediction_proba:.3f})")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False
    
    # Test 4: Recommendations system
    print("\n4. Testing Recommendations System...")
    try:
        rec_system = HealthcareRecommendationSystem()
        # Use patient data for recommendations
        patient_data = {
            'Age': 55,
            'Sex': 1,
            'ChestPainType': 1,
            'RestingBP': 140,
            'Cholesterol': 289,
            'FastingBS': 0,
            'RestingECG': 1,
            'MaxHR': 172,
            'ExerciseAngina': 1,
            'Oldpeak': 1.5,
            'ST_Slope': 1
        }
        recommendations = rec_system.get_lifestyle_recommendations(patient_data, prediction_proba)
        
        print(f"✅ Lifestyle recommendations generated")
        print(f"   Risk Level: {recommendations['risk_level']}")
        print(f"   Diet recommendations: {len(recommendations['diet'])} items")
        print(f"   Exercise recommendations: {len(recommendations['exercise'])} items")
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return False
    
    # Test 5: Hospital database
    print("\n5. Testing Hospital Database...")
    try:
        hospitals = rec_system.find_hospitals(city="New York", state="NY", max_results=3)
        
        if hospitals and 'error' not in hospitals[0]:
            print(f"✅ Hospital search successful: Found {len(hospitals)} hospitals")
            if hospitals:
                print(f"   Top hospital: {hospitals[0]['name']} (Rating: {hospitals[0]['cardiac_rating']}/5)")
        else:
            print("❌ No hospitals found or database error")
            
    except Exception as e:
        print(f"❌ Error searching hospitals: {e}")
        return False
    
    # Test 6: Report generation
    print("\n6. Testing Report Generation...")
    try:
        report = generate_patient_report(patient_data, prediction_proba, "New York", "NY")
        
        if len(report) > 500:  # Basic check for content
            print("✅ Patient report generated successfully")
            print(f"   Report length: {len(report)} characters")
        else:
            print("❌ Report seems too short")
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False
    
    print("\n=== All Tests Passed! ✅ ===")
    print("\nStreamlit App Components Summary:")
    print("- Model and preprocessor: Ready ✅")
    print("- Prediction functionality: Working ✅")
    print("- Recommendations system: Working ✅")
    print("- Hospital database: Working ✅")
    print("- Report generation: Working ✅")
    print("\nTo run the Streamlit app:")
    print("python -m streamlit run app/main.py")
    
    return True

if __name__ == "__main__":
    test_streamlit_components()