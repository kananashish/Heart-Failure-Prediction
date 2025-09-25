# User Guide

## Getting Started with Heart Failure Prediction System

This guide will help you get up and running with the Heart Failure Prediction System quickly and effectively.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using the Web Interface](#using-the-web-interface)
4. [Command Line Usage](#command-line-usage)
5. [Understanding Results](#understanding-results)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

## Installation

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: Version 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space

### Step-by-Step Installation

#### 1. Download the Project

**Option A: Clone from Git**
```bash
git clone https://github.com/your-username/heart-failure-prediction.git
cd heart-failure-prediction
```

**Option B: Download ZIP**
- Download the project ZIP file
- Extract to your desired location
- Open terminal/command prompt in the extracted folder

#### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv heart_failure_env
heart_failure_env\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv heart_failure_env
source heart_failure_env/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import streamlit, pandas, scikit_learn; print('Installation successful!')"
```

## Quick Start

### Launch the Web Application

1. **Start the application:**
   ```bash
   streamlit run app/main.py
   ```

2. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - The application should load automatically

3. **Make your first prediction:**
   - Enter patient data in the sidebar
   - Click "Predict Heart Failure Risk"
   - View results and recommendations

### Test with Sample Data

The system includes sample data for testing:

```bash
# Run a quick test
python -c "
from src.preprocess import HeartDiseasePreprocessor
preprocessor = HeartDiseasePreprocessor('data/heart.csv')
data = preprocessor.load_data()
print(f'Loaded {len(data)} patient records successfully!')
"
```

## Using the Web Interface

### Main Dashboard

The web interface has several key sections:

#### 1. **Sidebar: Patient Input**
- **Personal Information**: Age, gender, and basic demographics
- **Clinical Measurements**: Blood pressure, cholesterol, heart rate
- **Medical History**: Previous conditions and test results
- **Symptoms**: Current symptoms and pain types

#### 2. **Main Panel: Results**
- **Risk Assessment**: Heart failure probability with confidence
- **Risk Visualization**: Color-coded risk level display
- **Model Explanation**: Feature importance and SHAP values

#### 3. **Recommendations Tab**
- **Lifestyle Advice**: Personalized recommendations based on risk
- **Medical Care**: Specialist and hospital suggestions
- **Emergency Guidelines**: When to seek immediate care

#### 4. **Batch Processing Tab**
- **File Upload**: Process multiple patients from CSV
- **Results Download**: Export predictions and recommendations
- **Summary Statistics**: Overview of batch results

### Input Guidelines

#### Required Fields
- **Age**: Patient age in years (18-100)
- **Sex**: Male (M) or Female (F)
- **ChestPainType**: Select from dropdown options
- **RestingBP**: Resting blood pressure (80-200 mmHg)
- **Cholesterol**: Serum cholesterol (100-600 mg/dl)

#### Optional Fields
- **FastingBS**: Fasting blood sugar status
- **RestingECG**: Electrocardiogram results
- **MaxHR**: Maximum heart rate (60-220 bpm)
- **ExerciseAngina**: Exercise-induced chest pain
- **Oldpeak**: ST depression value
- **ST_Slope**: ST segment slope

### Understanding Input Options

#### Chest Pain Types
- **TA (Typical Angina)**: Classic chest pain related to heart issues
- **ATA (Atypical Angina)**: Chest pain with some unusual features
- **NAP (Non-Anginal Pain)**: Chest pain not related to heart
- **ASY (Asymptomatic)**: No chest pain symptoms

#### ECG Results
- **Normal**: Normal heart rhythm
- **ST**: ST-T wave abnormality
- **LVH**: Left ventricular hypertrophy

#### ST Slope
- **Up**: Upsloping ST segment (typically better)
- **Flat**: Flat ST segment
- **Down**: Downsloping ST segment (concerning)

## Command Line Usage

For advanced users and automation:

### Basic Prediction Script

Create a file `predict.py`:

```python
from src.preprocess import HeartDiseasePreprocessor
from src.train import ModelTrainer
import pandas as pd

# Sample patient data
patient = {
    'Age': 65,
    'Sex': 1,  # 1 for Male, 0 for Female
    'ChestPainType': 0,  # ASY
    'RestingBP': 140,
    'Cholesterol': 250,
    'FastingBS': 1,
    'RestingECG': 0,
    'MaxHR': 150,
    'ExerciseAngina': 1,
    'Oldpeak': 2.0,
    'ST_Slope': 1
}

# Load and train model (or load pre-trained)
preprocessor = HeartDiseasePreprocessor('data/heart.csv')
trainer = ModelTrainer()
trainer.initialize_models()

# Make prediction
patient_df = pd.DataFrame([patient])
X_processed = preprocessor.transform(patient_df)
risk_probability = trainer.best_model.predict_proba(X_processed)[0][1]

print(f"Heart Failure Risk: {risk_probability:.2%}")
```

### Batch Processing Script

```python
import pandas as pd
from src.preprocess import HeartDiseasePreprocessor
from src.train import ModelTrainer

# Load patient data
patients_df = pd.read_csv('patients_to_predict.csv')

# Process all patients
preprocessor = HeartDiseasePreprocessor()
trainer = ModelTrainer()

X_processed = preprocessor.transform(patients_df)
predictions = trainer.best_model.predict_proba(X_processed)[:, 1]

# Save results
results_df = patients_df.copy()
results_df['Heart_Failure_Risk'] = predictions
results_df.to_csv('prediction_results.csv', index=False)

print(f"Processed {len(patients_df)} patients")
print(f"Average risk: {predictions.mean():.2%}")
```

## Understanding Results

### Risk Levels

The system categorizes risk into four levels:

#### 🟢 Low Risk (0-25%)
- **Interpretation**: Very low probability of heart failure
- **Action**: Continue regular health maintenance
- **Follow-up**: Annual cardiac screening

#### 🟡 Moderate Risk (25-50%)
- **Interpretation**: Some risk factors present
- **Action**: Lifestyle modifications recommended
- **Follow-up**: Semi-annual check-ups

#### 🟠 High Risk (50-75%)
- **Interpretation**: Significant risk factors
- **Action**: Medical consultation recommended
- **Follow-up**: Quarterly monitoring

#### 🔴 Critical Risk (75-100%)
- **Interpretation**: High probability of heart failure
- **Action**: Immediate medical attention required
- **Follow-up**: Ongoing cardiac care

### Model Confidence

- **High Confidence (>90%)**: Very reliable prediction
- **Moderate Confidence (70-90%)**: Generally reliable
- **Low Confidence (<70%)**: Consider additional testing

### Feature Importance

The system shows which factors most influenced the prediction:

- **High Impact Features**: Major contributors to risk assessment
- **Moderate Impact**: Important but not decisive factors
- **Low Impact**: Minor contributors to the decision

## Advanced Features

### EDA (Exploratory Data Analysis)

Generate comprehensive data analysis reports:

```python
from src.eda_analysis import HeartFailureEDA

eda = HeartFailureEDA('data/heart.csv')
report = eda.generate_comprehensive_report()

# Generates:
# - Distribution plots
# - Correlation matrices  
# - Statistical summaries
# - Missing value analysis
```

### Fairness Analysis

Analyze model bias across different demographic groups:

```python
from src.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer()
fairness_report = analyzer.analyze_model_fairness(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    sensitive_features={'Age': age_groups, 'Sex': sex_values}
)
```

### Hospital Database Queries

Find nearby hospitals with cardiology services:

```python
from src.recommendations import HealthcareRecommendationSystem

rec_system = HealthcareRecommendationSystem()
hospitals = rec_system.find_nearby_hospitals(
    city='Houston',
    state='TX',
    limit=5
)

for hospital in hospitals:
    print(f"{hospital['name']} - Rating: {hospital['rating']}")
```

## File Formats

### CSV Input Format

For batch processing, use this CSV format:

```csv
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
65,1,0,140,250,1,0,150,1,2.0,1
45,0,1,120,200,0,0,180,0,1.0,2
```

### Column Encoding

- **Sex**: 1 = Male, 0 = Female  
- **ChestPainType**: 0 = ASY, 1 = ATA, 2 = NAP, 3 = TA
- **FastingBS**: 1 = >120 mg/dl, 0 = ≤120 mg/dl
- **RestingECG**: 0 = Normal, 1 = ST, 2 = LVH
- **ExerciseAngina**: 1 = Yes, 0 = No
- **ST_Slope**: 0 = Down, 1 = Flat, 2 = Up

## Troubleshooting

### Common Issues

#### 1. **"ModuleNotFoundError"**
```bash
# Solution: Activate virtual environment and install dependencies
source heart_failure_env/bin/activate  # or heart_failure_env\Scripts\activate on Windows
pip install -r requirements.txt
```

#### 2. **"FileNotFoundError: heart.csv"**
```bash
# Solution: Ensure data file is in correct location
ls data/heart.csv  # Check if file exists
# Download from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
```

#### 3. **"Port 8501 is already in use"**
```bash
# Solution: Use different port or kill existing process
streamlit run app/main.py --server.port 8502
```

#### 4. **Slow Performance**
- **Cause**: Large dataset or limited resources
- **Solution**: Reduce dataset size or increase system memory
- **Alternative**: Use batch processing with smaller chunks

#### 5. **Prediction Seems Wrong**
- **Check**: Input data format and units
- **Verify**: All required fields are filled
- **Consider**: Model confidence level

### Getting Help

1. **Check Documentation**: Review this guide and API documentation
2. **Examine Logs**: Look for error messages in terminal
3. **Test with Sample Data**: Use included test data to verify setup
4. **GitHub Issues**: Report bugs or request features
5. **Community**: Join discussions and share experiences

### Performance Optimization

#### For Large Datasets
```python
# Process in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    predictions = model.predict(preprocessor.transform(chunk))
    # Save chunk results
```

#### For Faster Loading
```python
# Cache preprocessor
import pickle
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Load cached version
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
```

## Best Practices

### Data Quality
- **Validate inputs** before prediction
- **Handle missing values** appropriately  
- **Check data ranges** for reasonableness
- **Use consistent units** (mmHg for BP, mg/dl for cholesterol)

### Interpretation
- **Consider confidence levels** when making decisions
- **Combine with clinical judgment** 
- **Look at feature importance** for insights
- **Use trends over time** rather than single predictions

### Medical Ethics
- **Never replace medical consultation**
- **Explain limitations** to users
- **Maintain patient privacy**
- **Document decision processes**

---

*This system is for educational and research purposes. Always consult healthcare professionals for medical decisions.*