# API Documentation

## Core Modules

### HeartDiseasePreprocessor

**Location**: `src/preprocess.py`

A comprehensive data preprocessing pipeline for heart failure prediction data.

#### Class: `HeartDiseasePreprocessor`

```python
class HeartDiseasePreprocessor:
    def __init__(self, data_path: str = None, target_column: str = 'HeartDisease')
```

**Parameters:**
- `data_path` (str, optional): Path to the CSV data file
- `target_column` (str, default='HeartDisease'): Name of the target column

#### Methods

##### `load_data() -> pd.DataFrame`
Loads data from CSV file or returns existing data.

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
preprocessor = HeartDiseasePreprocessor('data/heart.csv')
df = preprocessor.load_data()
```

##### `fit_transform(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]`
Fits preprocessing pipeline and transforms data.

**Parameters:**
- `data` (pd.DataFrame): Input dataset

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Processed features and target

**Example:**
```python
X_processed, y_processed = preprocessor.fit_transform(df)
```

##### `handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`
Handles missing values using median/mode imputation.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with missing values handled

##### `encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame`
Encodes categorical features using label encoding.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with encoded categorical features

##### `select_features(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame`
Performs feature selection using SelectKBest.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target vector
- `k` (int, default=10): Number of top features to select

**Returns:**
- `pd.DataFrame`: Selected features

##### `scale_features(X: pd.DataFrame) -> pd.DataFrame`
Scales features using StandardScaler.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix

**Returns:**
- `pd.DataFrame`: Scaled features

---

### ModelTrainer

**Location**: `src/train.py`

Comprehensive model training and evaluation system.

#### Class: `ModelTrainer`

```python
class ModelTrainer:
    def __init__(self)
```

#### Methods

##### `initialize_models() -> None`
Initializes multiple ML models with default parameters.

**Example:**
```python
trainer = ModelTrainer()
trainer.initialize_models()
```

##### `train_models(X_train, y_train, X_test, y_test) -> Dict`
Trains all initialized models and evaluates performance.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training target
- `X_test`: Testing features  
- `y_test`: Testing target

**Returns:**
- `Dict`: Training results and metrics

##### `hyperparameter_tuning(X_train, y_train, model_name: str = 'XGBoost') -> Dict`
Performs hyperparameter tuning using GridSearchCV.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training target
- `model_name` (str): Name of model to tune

**Returns:**
- `Dict`: Best parameters and performance metrics

##### `explain_model_predictions(model_name: str, X_train, X_test, max_display: int = 20) -> Dict`
Generates SHAP explanations for model predictions.

**Parameters:**
- `model_name` (str): Name of trained model
- `X_train`: Training features for SHAP explainer
- `X_test`: Test features to explain
- `max_display` (int): Maximum features to display

**Returns:**
- `Dict`: SHAP values and explanation plots

##### `get_best_model() -> Tuple[str, object]`
Returns the best performing model.

**Returns:**
- `Tuple[str, object]`: Best model name and model object

---

### HealthcareRecommendationSystem

**Location**: `src/recommendations.py`

Generates personalized healthcare recommendations.

#### Class: `HealthcareRecommendationSystem`

```python
class HealthcareRecommendationSystem:
    def __init__(self, hospital_db_path: str = 'data/hospitals.db')
```

#### Methods

##### `get_comprehensive_recommendations(prediction_prob: float, patient_data: Dict) -> Dict`
Generates comprehensive healthcare recommendations.

**Parameters:**
- `prediction_prob` (float): Heart failure risk probability (0-1)
- `patient_data` (Dict): Patient information dictionary

**Returns:**
- `Dict`: Comprehensive recommendations including lifestyle, specialists, hospitals

**Example:**
```python
rec_system = HealthcareRecommendationSystem()
recommendations = rec_system.get_comprehensive_recommendations(
    prediction_prob=0.75,
    patient_data={'Age': 65, 'Sex': 1}
)
```

##### `get_lifestyle_recommendations(risk_level: str, patient_data: Dict) -> Dict`
Provides lifestyle recommendations based on risk level.

**Parameters:**
- `risk_level` (str): 'low', 'moderate', or 'high'
- `patient_data` (Dict): Patient information

**Returns:**
- `Dict`: Lifestyle recommendations

##### `find_nearby_hospitals(city: str, state: str, limit: int = 10) -> List[Dict]`
Finds nearby hospitals with cardiology services.

**Parameters:**
- `city` (str): Patient's city
- `state` (str): Patient's state  
- `limit` (int): Maximum number of hospitals

**Returns:**
- `List[Dict]`: List of hospital information

---

### HeartFailureEDA

**Location**: `src/eda_analysis.py`

Automated exploratory data analysis system.

#### Class: `HeartFailureEDA`

```python
class HeartFailureEDA:
    def __init__(self, data_path: str)
```

#### Methods

##### `generate_comprehensive_report() -> Dict`
Generates complete EDA report with visualizations.

**Returns:**
- `Dict`: Analysis results and statistics

##### `create_distribution_plots() -> None`
Creates distribution plots for all features.

##### `create_correlation_analysis() -> Dict`
Analyzes feature correlations.

**Returns:**
- `Dict`: Correlation matrices and analysis

##### `analyze_missing_values() -> Dict`
Analyzes patterns in missing data.

**Returns:**
- `Dict`: Missing value analysis

##### `generate_statistical_summary() -> Dict`
Generates comprehensive statistical summary.

**Returns:**
- `Dict`: Statistical measures and insights

---

### FairnessAnalyzer

**Location**: `src/fairness.py`

Analyzes model bias and fairness metrics.

#### Class: `FairnessAnalyzer`

```python
class FairnessAnalyzer:
    def __init__(self)
```

#### Methods

##### `analyze_model_fairness(model, X_test, y_test, sensitive_features: Dict) -> Dict`
Performs comprehensive fairness analysis.

**Parameters:**
- `model`: Trained ML model
- `X_test`: Test features
- `y_test`: Test labels
- `sensitive_features` (Dict): Dictionary of sensitive attributes

**Returns:**
- `Dict`: Fairness metrics and analysis

##### `calculate_demographic_parity(y_true, y_pred, sensitive_attribute) -> float`
Calculates demographic parity metric.

##### `calculate_equalized_odds(y_true, y_pred, sensitive_attribute) -> float`
Calculates equalized odds metric.

##### `generate_bias_report(fairness_metrics: Dict) -> str`
Generates comprehensive bias analysis report.

---

## Database Functions

### Hospital Database

**Location**: `src/database.py`

Functions for hospital database management.

#### `create_synthetic_hospital_data(n_hospitals: int = 200) -> pd.DataFrame`
Creates synthetic hospital dataset.

**Parameters:**
- `n_hospitals` (int): Number of hospitals to generate

**Returns:**
- `pd.DataFrame`: Hospital dataset

#### `create_sqlite_database(df: pd.DataFrame, db_path: str) -> None`
Creates SQLite database from DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Hospital data
- `db_path` (str): Path for database file

#### `query_hospitals_by_location(db_path: str, city: str = None, state: str = None) -> List[Tuple]`
Queries hospitals by location.

**Parameters:**
- `db_path` (str): Path to database
- `city` (str, optional): City filter
- `state` (str, optional): State filter

**Returns:**
- `List[Tuple]`: Hospital query results

---

## Error Handling

All modules implement comprehensive error handling:

- **FileNotFoundError**: Raised when data files are not found
- **ValueError**: Raised for invalid input parameters
- **KeyError**: Raised when required columns are missing
- **AttributeError**: Raised when methods are not implemented

## Configuration

### Environment Variables

- `HEART_FAILURE_DB_PATH`: Path to hospital database (default: 'data/hospitals.db')
- `MODEL_CACHE_DIR`: Directory for cached models (default: 'models/')
- `LOG_LEVEL`: Logging level (default: 'INFO')

### Model Parameters

Default hyperparameters are defined in each model's initialization. Override using hyperparameter tuning methods.

## Examples

### Complete Workflow Example

```python
from src.preprocess import HeartDiseasePreprocessor
from src.train import ModelTrainer
from src.recommendations import HealthcareRecommendationSystem

# 1. Data preprocessing
preprocessor = HeartDiseasePreprocessor('data/heart.csv')
data = preprocessor.load_data()
X_processed, y_processed = preprocessor.fit_transform(data)

# 2. Model training
trainer = ModelTrainer()
trainer.initialize_models()
results = trainer.train_models(X_train, y_train, X_test, y_test)

# 3. Get recommendations
rec_system = HealthcareRecommendationSystem()
recommendations = rec_system.get_comprehensive_recommendations(
    prediction_prob=0.8,
    patient_data={'Age': 65, 'Sex': 1}
)
```

### Batch Prediction Example

```python
import pandas as pd

# Load new patient data
new_patients = pd.read_csv('new_patients.csv')

# Preprocess
X_new = preprocessor.transform(new_patients)

# Predict
best_model = trainer.get_best_model()[1]
predictions = best_model.predict_proba(X_new)[:, 1]

# Generate recommendations for each patient
for i, prob in enumerate(predictions):
    patient_rec = rec_system.get_comprehensive_recommendations(
        prediction_prob=prob,
        patient_data=new_patients.iloc[i].to_dict()
    )
    print(f"Patient {i+1}: Risk {prob:.2%}")
    print(patient_rec['summary'])
```