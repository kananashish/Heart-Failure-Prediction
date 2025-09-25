"""
Quick test to verify model loading works without import errors
"""
import os
import sys
import joblib
import pandas as pd

# Add the same path setup as in main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

def simple_preprocess_data(data):
    """Simple preprocessing function that mimics the original preprocessor without requiring the pickled class."""
    # If it's not already a DataFrame, convert it
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame([data])
    
    # Create a copy to avoid modifying original
    processed_data = data.copy()
    
    # Handle the expected feature columns based on the model input format
    expected_features = [
        'Age', 'Sex', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
        'ChestPainType_ATA', 'ChestPainType_NAP', 'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0.0
    
    # Return only the expected features in the right order
    return processed_data[expected_features]

def test_model_loading():
    """Test if model can be loaded without errors"""
    try:
        # Try models directory first
        model_path = os.path.join('models', 'best_heart_model.pkl')
        
        # If not found, try data directory
        if not os.path.exists(model_path):
            model_path = os.path.join('data', 'best_model.pkl')
        
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            
            model = joblib.load(model_path)
            
            # Handle case where model is stored as dictionary
            if isinstance(model, dict):
                print("Model is a dictionary with keys:", list(model.keys()))
                if 'model' in model:
                    model = model['model']
                elif 'best_model' in model:
                    model = model['best_model']
            
            print(f"Model type: {type(model)}")
            print("✅ Model loaded successfully!")
            
            # Test a simple prediction
            test_data = {
                'Age': 50, 'Sex': 1, 'FastingBS': 0, 'MaxHR': 150, 'ExerciseAngina': 0, 'Oldpeak': 1.0,
                'ChestPainType_ATA': 0, 'ChestPainType_NAP': 1, 'ST_Slope_Flat': 1, 'ST_Slope_Up': 0
            }
            
            processed_data = simple_preprocess_data(test_data)
            print(f"Processed data shape: {processed_data.shape}")
            
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)
            
            print(f"Test prediction: {prediction[0]}")
            print(f"Test prediction probability: {prediction_proba[0]}")
            print("✅ Prediction test successful!")
            
            return True
        else:
            print("❌ Model files not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading and prediction test PASSED!")
    else:
        print("\n💥 Model loading test FAILED!")