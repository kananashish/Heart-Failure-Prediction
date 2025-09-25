"""
Streamlit web application for heart failure prediction.
Provides interactive interface for predictions, visualizations, and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from typing import Dict, List, Tuple
import sqlite3
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import HeartDiseasePreprocessor
from recommendations import HealthcareRecommendationSystem, generate_patient_report

def create_model_input(patient_data):
    """Convert raw patient data to model input format."""
    # Map categorical values to encoded format
    chest_pain_mapping = {0: [1, 0], 1: [0, 1], 2: [0, 0], 3: [0, 0]}  # ATA, NAP, TA, ASY
    st_slope_mapping = {0: [0, 1], 1: [1, 0], 2: [0, 0]}  # Up, Flat, Down
    
    chest_pain_encoded = chest_pain_mapping.get(patient_data['ChestPainType'], [0, 0])
    st_slope_encoded = st_slope_mapping.get(patient_data['ST_Slope'], [0, 0])
    
    model_input = {
        'Age': float(patient_data['Age']),
        'Sex': float(patient_data['Sex']),
        'FastingBS': float(patient_data['FastingBS']),
        'MaxHR': float(patient_data['MaxHR']),
        'ExerciseAngina': float(patient_data['ExerciseAngina']),
        'Oldpeak': float(patient_data['Oldpeak']),
        'ChestPainType_ATA': float(chest_pain_encoded[0]),
        'ChestPainType_NAP': float(chest_pain_encoded[1]),
        'ST_Slope_Flat': float(st_slope_encoded[0]),
        'ST_Slope_Up': float(st_slope_encoded[1])
    }
    
    return pd.DataFrame([model_input])

# Configure Streamlit page
st.set_page_config(
    page_title="Heart Failure Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-moderate {
        background-color: #fff8e1;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .recommendation-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_heart_model.pkl')
        preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl')
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            return model, preprocessor
        else:
            st.error("Model files not found. Please train the model first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df.head(1000)  # Limit for demo
        else:
            # Create sample data if file doesn't exist
            sample_data = {
                'Age': [54, 37, 41, 56, 57],
                'Sex': [1, 1, 0, 1, 0],
                'ChestPainType': [2, 1, 1, 0, 0],
                'RestingBP': [150, 130, 140, 120, 120],
                'Cholesterol': [195, 250, 204, 236, 354],
                'FastingBS': [0, 0, 0, 0, 0],
                'RestingECG': [1, 1, 0, 1, 1],
                'MaxHR': [122, 187, 172, 178, 163],
                'ExerciseAngina': [0, 0, 0, 0, 1],
                'Oldpeak': [0, 3.5, 1.4, 0.8, 0.6],
                'ST_Slope': [2, 0, 2, 2, 2],
                'HeartDisease': [0, 0, 0, 0, 0]
            }
            return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_prediction_interface():
    """Create the main prediction interface."""
    st.markdown('<h1 class="main-header">❤️ Heart Failure Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        st.error("Unable to load model. Please ensure the model is trained first.")
        return
    
    # Sidebar for input
    st.sidebar.header("Patient Information")
    
    # Patient demographics
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", min_value=20, max_value=100, value=50, step=1)
    sex = st.sidebar.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    
    # Clinical measurements
    st.sidebar.subheader("Clinical Measurements")
    chest_pain_type = st.sidebar.selectbox(
        "Chest Pain Type",
        options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-Anginal", 2), ("Asymptomatic", 3)],
        format_func=lambda x: x[0]
    )
    
    resting_bp = st.sidebar.slider("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
    cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", min_value=0, max_value=500, value=200, step=1)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    
    # ECG and exercise data
    st.sidebar.subheader("ECG & Exercise Data")
    resting_ecg = st.sidebar.selectbox(
        "Resting ECG",
        options=[("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)],
        format_func=lambda x: x[0]
    )
    
    max_hr = st.sidebar.slider("Maximum Heart Rate", min_value=60, max_value=220, value=150, step=1)
    exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    st_slope = st.sidebar.selectbox(
        "ST Slope",
        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        format_func=lambda x: x[0]
    )
    
    # Location for hospital recommendations
    st.sidebar.subheader("Location (Optional)")
    city = st.sidebar.text_input("City", placeholder="e.g., New York")
    state = st.sidebar.text_input("State", placeholder="e.g., NY")
    
    # Create patient data dictionary
    patient_data = {
        'Age': age,
        'Sex': sex[1],
        'ChestPainType': chest_pain_type[1],
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs[1],
        'RestingECG': resting_ecg[1],
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina[1],
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope[1]
    }
    
    # Prediction button
    if st.sidebar.button("🔍 Predict Heart Disease Risk", type="primary"):
        # Make prediction
        try:
            # Prepare data for prediction
            input_df = pd.DataFrame([patient_data])
            
            # Make prediction using the actual estimator
            estimator = model['model'] if isinstance(model, dict) else model
            
            # Create input with selected features format
            processed_input = create_model_input(patient_data)
            
            prediction_proba = estimator.predict_proba(processed_input)[0][1]  # Probability of heart disease
            prediction = estimator.predict(processed_input)[0]
            
            # Display results in main area
            display_prediction_results(patient_data, prediction, prediction_proba, city, state)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def display_prediction_results(patient_data: Dict, prediction: int, prediction_proba: float, 
                             city: str = None, state: str = None):
    """Display prediction results and recommendations."""
    
    # Initialize recommendation system
    rec_system = HealthcareRecommendationSystem()
    risk_level = rec_system.assess_risk_level(prediction_proba)
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Risk level display
        risk_color = {"low": "#4caf50", "moderate": "#ff9800", "high": "#f44336"}
        risk_class = f"risk-{risk_level}"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h2 style="color: {risk_color[risk_level]}; text-align: center; margin-bottom: 1rem;">
                {risk_level.upper()} RISK
            </h2>
            <h3 style="text-align: center; margin-bottom: 0.5rem;">
                Prediction Probability: {prediction_proba:.2%}
            </h3>
            <h4 style="text-align: center; margin-top: 0;">
                Classification: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}
            </h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "💡 Recommendations", "🏥 Healthcare", "📋 Full Report"])
    
    with tab1:
        display_analysis_tab(patient_data, prediction_proba, risk_level)
    
    with tab2:
        display_recommendations_tab(patient_data, prediction_proba)
    
    with tab3:
        display_healthcare_tab(city, state, risk_level)
    
    with tab4:
        display_report_tab(patient_data, prediction_proba, city, state)

def display_analysis_tab(patient_data: Dict, prediction_proba: float, risk_level: str):
    """Display analysis and visualizations."""
    st.subheader("📊 Risk Analysis")
    
    # Risk meter
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Patient characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Patient Profile")
        st.markdown(f"**Age:** {patient_data['Age']} years")
        st.markdown(f"**Sex:** {'Male' if patient_data['Sex'] == 1 else 'Female'}")
        st.markdown(f"**Resting BP:** {patient_data['RestingBP']} mmHg")
        st.markdown(f"**Cholesterol:** {patient_data['Cholesterol']} mg/dl")
        st.markdown(f"**Max Heart Rate:** {patient_data['MaxHR']} bpm")
        
    with col2:
        st.markdown("### Risk Factors")
        
        # Create risk factor visualization
        risk_factors = []
        risk_values = []
        
        if patient_data['Age'] > 60:
            risk_factors.append("Advanced Age")
            risk_values.append(min(100, (patient_data['Age'] - 40) * 2))
        
        if patient_data['RestingBP'] > 140:
            risk_factors.append("High Blood Pressure")
            risk_values.append(min(100, (patient_data['RestingBP'] - 120) * 2))
        
        if patient_data['Cholesterol'] > 200:
            risk_factors.append("High Cholesterol")
            risk_values.append(min(100, (patient_data['Cholesterol'] - 200) * 0.5))
        
        if patient_data['ExerciseAngina'] == 1:
            risk_factors.append("Exercise Angina")
            risk_values.append(80)
        
        if patient_data['MaxHR'] < 120:
            risk_factors.append("Low Max Heart Rate")
            risk_values.append(min(100, (140 - patient_data['MaxHR']) * 2))
        
        if risk_factors:
            fig_risk = px.bar(
                x=risk_values,
                y=risk_factors,
                orientation='h',
                title="Risk Factor Severity",
                color=risk_values,
                color_continuous_scale="Reds"
            )
            fig_risk.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.success("No major risk factors detected!")

def display_recommendations_tab(patient_data: Dict, prediction_proba: float):
    """Display lifestyle recommendations."""
    st.subheader("💡 Personalized Recommendations")
    
    # Get recommendations
    rec_system = HealthcareRecommendationSystem()
    recommendations = rec_system.get_lifestyle_recommendations(patient_data, prediction_proba)
    
    # Display recommendations in expandable sections
    with st.expander("🍎 Dietary Recommendations", expanded=True):
        for rec in recommendations['diet']:
            st.markdown(f"• {rec}")
    
    with st.expander("🏃 Exercise Guidelines", expanded=True):
        for rec in recommendations['exercise']:
            st.markdown(f"• {rec}")
    
    with st.expander("🌱 Lifestyle Changes", expanded=True):
        for rec in recommendations['lifestyle']:
            st.markdown(f"• {rec}")
    
    with st.expander("📱 Monitoring Requirements", expanded=True):
        for rec in recommendations['monitoring']:
            st.markdown(f"• {rec}")
    
    with st.expander("🚨 Emergency Warning Signs", expanded=False):
        st.error("Call 911 immediately if you experience:")
        for sign in recommendations['emergency_signs']:
            st.markdown(f"• {sign}")

def display_healthcare_tab(city: str, state: str, risk_level: str):
    """Display healthcare provider recommendations."""
    st.subheader("🏥 Healthcare Providers")
    
    # Get hospital recommendations
    rec_system = HealthcareRecommendationSystem()
    
    # Find hospitals
    emergency = (risk_level == 'high')
    hospitals = rec_system.find_hospitals(city, state, emergency=emergency, max_results=10)
    
    if hospitals and 'error' not in hospitals[0]:
        st.success(f"Found {len(hospitals)} recommended healthcare facilities")
        
        for i, hospital in enumerate(hospitals[:5], 1):
            with st.expander(f"{i}. {hospital['name']} ⭐ {hospital['cardiac_rating']}/5.0"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Address:** {hospital['address']}")
                    st.markdown(f"**City:** {hospital['city']}, {hospital['state']}")
                    st.markdown(f"**Phone:** {hospital['phone']}")
                    
                with col2:
                    st.markdown(f"**Cardiac Rating:** {hospital['cardiac_rating']}/5.0")
                    st.markdown(f"**Emergency Services:** {'✅ Yes' if hospital['emergency_services'] else '❌ No'}")
                    st.markdown(f"**Bed Capacity:** {hospital['beds_count']}")
                    
                st.markdown(f"**Specializations:** {hospital['specializations']}")
                
                if hospital['website']:
                    st.markdown(f"**Website:** [{hospital['website']}]({hospital['website']})")
    
    else:
        st.warning("No hospitals found for the specified location. Showing general recommendations:")
        st.info("Try searching without city/state filters, or check nearby major cities.")
    
    # Specialist recommendations
    st.subheader("👨‍⚕️ Specialist Recommendations")
    specialists = rec_system.get_specialist_recommendations(risk_level, {})
    
    if specialists['primary']:
        st.markdown("#### Primary Specialists")
        for spec in specialists['primary']:
            st.markdown(f"**{spec['type']}** - {spec['reason']} (Every {spec['frequency']})")
    
    if specialists['secondary']:
        with st.expander("Additional Specialists"):
            for spec in specialists['secondary']:
                st.markdown(f"**{spec['type']}** - {spec['reason']} ({spec['frequency']})")

def display_report_tab(patient_data: Dict, prediction_proba: float, city: str, state: str):
    """Display full patient report."""
    st.subheader("📋 Comprehensive Patient Report")
    
    # Generate full report
    report = generate_patient_report(patient_data, prediction_proba, city, state)
    
    # Display report
    st.markdown(report)
    
    # Download button
    st.download_button(
        label="📥 Download Report as Markdown",
        data=report,
        file_name=f"heart_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def create_batch_prediction_interface():
    """Create interface for batch predictions."""
    st.header("📊 Batch Prediction Analysis")
    
    # Load sample data
    sample_data = load_sample_data()
    if sample_data is None:
        st.error("Unable to load sample data.")
        return
    
    # File upload option
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} records from uploaded file")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            df = sample_data
    else:
        df = sample_data
        st.info("Using sample data for demonstration")
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Load model for batch prediction
    model, preprocessor = load_model_and_preprocessor()
    if model is None:
        st.error("Model not available for batch prediction")
        return
    
    if st.button("Run Batch Predictions"):
        try:
            # Get the actual estimator from the model dictionary
            estimator = model['model'] if isinstance(model, dict) else model
            
            # Prepare features (remove target if present)
            feature_cols = [col for col in df.columns if col != 'HeartDisease']
            X = df[feature_cols].copy()
            
            # Convert categorical data to the format expected by the model
            processed_rows = []
            for idx, row in X.iterrows():
                # Convert each row to the format expected by create_model_input
                patient_data = {
                    'Age': row['Age'],
                    'Sex': 1 if row['Sex'] == 'M' else 0,  # M=1, F=0
                    'ChestPainType': {'ATA': 0, 'NAP': 1, 'TA': 2, 'ASY': 3}.get(row['ChestPainType'], 3),
                    'FastingBS': row['FastingBS'],
                    'MaxHR': row['MaxHR'],
                    'ExerciseAngina': 1 if row['ExerciseAngina'] == 'Y' else 0,  # Y=1, N=0
                    'Oldpeak': row['Oldpeak'],
                    'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}.get(row['ST_Slope'], 2)
                }
                
                # Use create_model_input to get the proper format
                model_input = create_model_input(patient_data)
                processed_rows.append(model_input.iloc[0])
            
            # Create DataFrame from processed rows
            X_processed = pd.DataFrame(processed_rows)
            
            # Make predictions
            predictions = estimator.predict(X_processed)
            prediction_probas = estimator.predict_proba(X_processed)[:, 1]
            
            # Add predictions to dataframe
            results_df = df.copy()
            results_df['Predicted_Risk'] = predictions
            results_df['Risk_Probability'] = prediction_probas
            results_df['Risk_Level'] = results_df['Risk_Probability'].apply(
                lambda x: 'High' if x > 0.7 else 'Moderate' if x > 0.3 else 'Low'
            )
            
            # Display summary statistics
            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Patients", len(results_df))
            with col2:
                high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
                st.metric("High Risk Patients", high_risk_count)
            with col3:
                avg_risk = results_df['Risk_Probability'].mean()
                st.metric("Average Risk", f"{avg_risk:.2%}")
            
            # Risk distribution
            fig_dist = px.histogram(
                results_df, 
                x='Risk_Probability', 
                nbins=20,
                title="Risk Probability Distribution",
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Risk level pie chart
            risk_counts = results_df['Risk_Level'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Display results
            st.subheader("Detailed Results")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during batch prediction: {e}")

def create_dashboard():
    """Create analytics dashboard."""
    st.header("📈 Analytics Dashboard")
    
    # Load sample data for analytics
    sample_data = load_sample_data()
    if sample_data is None:
        st.error("Unable to load data for dashboard.")
        return
    
    df = sample_data
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if 'HeartDisease' in df.columns:
            positive_cases = df['HeartDisease'].sum()
            st.metric("Positive Cases", positive_cases)
        else:
            st.metric("Positive Cases", "N/A")
    with col3:
        st.metric("Features", len(df.columns))
    with col4:
        missing_data = df.isnull().sum().sum()
        st.metric("Missing Values", missing_data)
    
    # Feature distributions
    st.subheader("Feature Analysis")
    
    # Select feature to analyze
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_feature = st.selectbox("Select feature to analyze:", numeric_cols)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig_hist = px.histogram(
                df, 
                x=selected_feature, 
                nbins=20,
                title=f"{selected_feature} Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by target (if available)
            if 'HeartDisease' in df.columns:
                fig_box = px.box(
                    df, 
                    x='HeartDisease', 
                    y=selected_feature,
                    title=f"{selected_feature} by Heart Disease Status"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                # Just show basic stats
                st.subheader(f"{selected_feature} Statistics")
                st.dataframe(df[selected_feature].describe())
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        st.subheader("Feature Correlations")
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🔍 Single Prediction", "📊 Batch Analysis", "📈 Dashboard"]
    )
    
    # Route to appropriate page
    if page == "🔍 Single Prediction":
        create_prediction_interface()
    elif page == "📊 Batch Analysis":
        create_batch_prediction_interface()
    elif page == "📈 Dashboard":
        create_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        ❤️ Heart Failure Prediction System | Built with Streamlit<br>
        <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Always consult healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()