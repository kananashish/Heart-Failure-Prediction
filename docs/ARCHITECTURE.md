# Technical Architecture Documentation

## System Overview

The Heart Failure Prediction System is built using a modular, scalable architecture that separates concerns and provides clear interfaces between components. The system follows modern software engineering practices with comprehensive testing, documentation, and deployment strategies.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │  API Gateway    │    │  ML Pipeline    │
│   (Streamlit)   │◄──►│  (FastAPI)      │◄──►│  (scikit-learn) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Static Files  │    │  Configuration  │    │  Model Storage  │
│   (CSS/JS)      │    │  (YAML/JSON)    │    │  (Pickle/ONNX)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Data Layer    │
                    │  (SQLite/CSV)   │
                    └─────────────────┘
```

## Core Components

### 1. Data Processing Layer (`src/`)

#### Preprocessing Pipeline (`src/preprocess.py`)
- **Purpose**: Clean, validate, and transform raw data
- **Components**:
  - `HeartDiseasePreprocessor`: Main preprocessing class
  - Missing value imputation using median/mode strategies
  - Categorical encoding with label encoders
  - Feature scaling with StandardScaler
  - Feature selection using SelectKBest

**Data Flow:**
```
Raw CSV → Load → Clean → Encode → Scale → Select Features → Processed Data
```

#### Data Preparation (`src/data_prep.py`)
- **Purpose**: Advanced data preparation utilities
- **Components**:
  - SMOTE integration for class balancing
  - Train-test splitting with stratification
  - Data validation and quality checks

### 2. Machine Learning Pipeline (`src/train.py`)

#### ModelTrainer Architecture
```python
class ModelTrainer:
    ├── initialize_models()      # Model instantiation
    ├── train_models()          # Training pipeline
    ├── hyperparameter_tuning() # Grid search optimization
    ├── cross_validate()        # K-fold validation
    ├── explain_predictions()   # SHAP interpretability
    └── save_models()           # Model persistence
```

#### Supported Algorithms
- **Tree-based**: Random Forest, XGBoost, CatBoost, Gradient Boosting
- **Linear**: Logistic Regression, SVM
- **Probabilistic**: Naive Bayes
- **Ensemble**: Voting classifiers

#### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Precision-Recall AUC  
- Confusion Matrix
- Cross-validation scores

### 3. Fairness and Bias Analysis (`src/fairness.py`)

#### FairnessAnalyzer Components
```python
├── Demographic Parity Assessment
├── Equalized Odds Calculation
├── Equal Opportunity Analysis
├── Calibration Metrics
├── Bias Visualization
└── Comprehensive Reporting
```

#### Fairness Metrics Implementation
- **Statistical Parity**: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
- **Equalized Odds**: TPR and FPR equality across groups
- **Equal Opportunity**: TPR equality for positive class
- **Calibration**: P(Y=1|Ŷ=p,A=a) consistency

### 4. Healthcare Recommendations (`src/recommendations.py`)

#### Recommendation Engine Architecture
```
Risk Assessment → Lifestyle Advice → Medical Referrals → Hospital Suggestions
       │               │                    │                   │
       ▼               ▼                    ▼                   ▼
  Risk Levels    Personalized         Specialist         Location-based
  (Low-Critical)   Guidelines         Matching           Hospital Search
```

#### Components
- **Risk Stratification**: 4-tier risk classification
- **Lifestyle Engine**: Evidence-based recommendations
- **Provider Network**: Hospital and specialist database
- **Emergency Protocols**: Critical care pathways

### 5. Database Layer (`src/database.py`)

#### Hospital Database Schema
```sql
CREATE TABLE hospitals (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    phone TEXT,
    cardiac_rating REAL,
    emergency_services BOOLEAN,
    specializations TEXT,
    beds_count INTEGER,
    website TEXT
);
```

#### Database Operations
- **Synthetic Data Generation**: Realistic hospital data creation
- **Location-based Queries**: Spatial search functionality
- **Rating Systems**: Quality metrics integration
- **Contact Management**: Provider information storage

### 6. Web Application Layer (`app/`)

#### Streamlit Architecture
```python
app/main.py
├── Sidebar: Patient Input Forms
├── Main Panel: Prediction Results
├── Recommendations Tab: Healthcare Advice
├── Batch Processing: CSV Upload/Download
├── Visualization: Interactive Charts
└── Export: PDF Reports
```

#### UI Components
- **Input Validation**: Real-time form validation
- **Progress Indicators**: Loading states and progress bars
- **Interactive Plots**: Plotly-based visualizations
- **Responsive Design**: Mobile-friendly interface

### 7. Automated EDA (`src/eda_analysis.py`)

#### HeartFailureEDA Architecture
```python
├── Data Quality Assessment
│   ├── Missing Value Analysis
│   ├── Outlier Detection
│   └── Data Type Validation
├── Statistical Analysis
│   ├── Descriptive Statistics
│   ├── Distribution Analysis
│   └── Correlation Matrices
├── Visualization Engine
│   ├── Univariate Plots
│   ├── Bivariate Analysis
│   └── Interactive Dashboards
└── Report Generation
    ├── HTML Reports
    ├── PDF Exports
    └── Summary Statistics
```

## Data Models

### Patient Data Model
```python
@dataclass
class PatientData:
    age: int
    sex: int  # 0: Female, 1: Male
    chest_pain_type: int  # 0-3 encoded
    resting_bp: float
    cholesterol: float
    fasting_bs: int  # 0 or 1
    resting_ecg: int  # 0-2 encoded
    max_hr: float
    exercise_angina: int  # 0 or 1
    oldpeak: float
    st_slope: int  # 0-2 encoded
```

### Prediction Result Model
```python
@dataclass
class PredictionResult:
    patient_id: str
    risk_probability: float
    risk_level: str
    confidence: float
    model_used: str
    timestamp: datetime
    feature_importance: Dict[str, float]
    shap_values: List[float]
```

### Hospital Model
```python
@dataclass
class Hospital:
    id: int
    name: str
    address: str
    city: str
    state: str
    phone: str
    rating: float
    cardiac_services: bool
    emergency_services: bool
    specializations: List[str]
```

## Design Patterns

### 1. Strategy Pattern
Used for model selection and algorithm switching:
```python
class ModelStrategy:
    def train(self, X, y): pass
    def predict(self, X): pass

class RandomForestStrategy(ModelStrategy): pass
class XGBoostStrategy(ModelStrategy): pass
```

### 2. Factory Pattern
Model instantiation and configuration:
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs):
        if model_type == 'rf':
            return RandomForestClassifier(**kwargs)
        elif model_type == 'xgb':
            return XGBClassifier(**kwargs)
```

### 3. Observer Pattern
Model training progress monitoring:
```python
class TrainingObserver:
    def on_epoch_end(self, epoch, metrics): pass
    def on_training_complete(self, final_metrics): pass
```

### 4. Pipeline Pattern
Data processing workflows:
```python
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('encoder', LabelEncoder()),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest())
])
```

## Configuration Management

### Environment Configuration
```yaml
# config/development.yaml
database:
  path: "data/hospitals.db"
  backup_path: "backups/"

models:
  cache_dir: "models/"
  auto_retrain: true
  performance_threshold: 0.85

logging:
  level: "DEBUG"
  file: "logs/app.log"
  max_size: "10MB"

api:
  host: "localhost"
  port: 8501
  debug: true
```

### Model Configuration
```yaml
# config/models.yaml
random_forest:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  random_state: 42

xgboost:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100
  objective: "binary:logistic"

hyperparameter_tuning:
  cv_folds: 5
  scoring: "roc_auc"
  n_jobs: -1
```

## Testing Architecture

### Testing Pyramid
```
    ┌──────────────┐
    │ E2E Tests    │ (5%)
    │ Integration  │
    └──────────────┘
   ┌────────────────┐
   │ Integration    │ (15%)
   │ Tests          │
   └────────────────┘
  ┌──────────────────┐
  │ Unit Tests       │ (80%)
  └──────────────────┘
```

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Scalability and speed testing
- **Security Tests**: Vulnerability assessment

### Test Configuration (`pytest.ini`)
```ini
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Long-running tests

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = 
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
```

## Security Considerations

### Data Privacy
- **No PII Storage**: Only aggregate/anonymized data
- **Secure Transmission**: HTTPS for all communications
- **Access Control**: Role-based permissions
- **Audit Logging**: Comprehensive activity logging

### Input Validation
```python
def validate_patient_data(data: Dict) -> bool:
    """Validates patient input data for security and correctness."""
    validators = {
        'age': lambda x: 0 < x < 150,
        'resting_bp': lambda x: 50 < x < 250,
        'cholesterol': lambda x: 0 <= x <= 1000,
        'max_hr': lambda x: 50 < x < 250
    }
    
    for field, validator in validators.items():
        if field in data and not validator(data[field]):
            return False
    return True
```

### Model Security
- **Input Sanitization**: Prevent injection attacks
- **Model Versioning**: Track and validate model versions
- **Adversarial Robustness**: Detect adversarial inputs
- **Output Validation**: Ensure predictions are within expected ranges

## Performance Optimization

### Caching Strategy
```python
# Model caching
@lru_cache(maxsize=128)
def load_model(model_path: str):
    return pickle.load(open(model_path, 'rb'))

# Data caching
@cached(TTLCache(maxsize=100, ttl=3600))
def preprocess_data(data_hash: str):
    return preprocessor.transform(data)
```

### Database Optimization
```sql
-- Indexed queries for hospital search
CREATE INDEX idx_hospital_location ON hospitals(city, state);
CREATE INDEX idx_hospital_rating ON hospitals(cardiac_rating);
CREATE INDEX idx_hospital_services ON hospitals(emergency_services);
```

### Memory Management
```python
# Efficient data loading
def load_large_dataset(file_path: str, chunk_size: int = 1000):
    """Load large datasets in chunks to manage memory."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk
```

## Monitoring and Logging

### Application Monitoring
```python
import logging
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.info(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper
```

### Health Checks
```python
def health_check():
    """System health monitoring."""
    checks = {
        'database': check_database_connection(),
        'models': check_model_availability(),
        'memory': check_memory_usage(),
        'disk': check_disk_space()
    }
    return all(checks.values()), checks
```

## Deployment Architecture

### Container Strategy
```dockerfile
# Multi-stage build
FROM python:3.9-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py"]
```

### Environment Separation
```
Development → Testing → Staging → Production
     ↓           ↓        ↓          ↓
  localhost   CI/CD    preview    live
```

## Scalability Considerations

### Horizontal Scaling
- **Load Balancing**: Multiple app instances
- **Database Sharding**: Distributed data storage
- **Microservices**: Component separation
- **API Gateway**: Request routing and management

### Vertical Scaling
- **Resource Optimization**: Memory and CPU tuning
- **Caching Layers**: Redis/Memcached integration
- **Database Optimization**: Query performance tuning
- **Model Optimization**: Quantization and pruning

## Future Architecture Enhancements

### Planned Improvements
1. **Microservices Architecture**: Separate prediction, recommendation, and database services
2. **Event-Driven Design**: Asynchronous processing with message queues
3. **GraphQL API**: Flexible data querying interface
4. **Real-time Streaming**: Live data processing capabilities
5. **ML Pipeline Orchestration**: Airflow/Kubeflow integration

### Technology Roadmap
- **Backend**: FastAPI migration for better performance
- **Database**: PostgreSQL for better scalability
- **Caching**: Redis for session management
- **Monitoring**: Prometheus/Grafana stack
- **Container Orchestration**: Kubernetes deployment

---

This technical architecture provides a solid foundation for a scalable, maintainable, and secure healthcare AI system while maintaining flexibility for future enhancements.