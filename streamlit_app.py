"""
Entry point for Streamlit Cloud deployment.
This file is required at the root level for Streamlit Cloud to recognize the app.
Imports and runs the main Streamlit application from app/main.py
"""

import streamlit as st
import os
import sys

# Add the app directory to the path
app_dir = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_dir)

# Add src directory to path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

# Configure page first
st.set_page_config(
    page_title="Heart Failure Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import and run the main app
try:
    from main import create_prediction_interface, simple_preprocess_data, load_model_and_preprocessor
    
    # Run the main application
    create_prediction_interface()
    
except ImportError as e:
    st.error(f"Failed to load application modules: {str(e)}")
    st.info("Please ensure all required modules are in the 'src' directory.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.text(traceback.format_exc())
