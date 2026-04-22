# Heart Failure Prediction Project Summary

## 1. What this project is
This project is an end-to-end machine learning system that predicts the risk of heart disease (heart failure risk) from clinical features. It includes:
- A data preprocessing and model training pipeline.
- Model evaluation with standard metrics and cross-validation.
- Fairness and bias analysis.
- A recommendation engine that provides lifestyle and healthcare guidance.
- A Streamlit web application for interactive predictions, batch processing, and reporting.
- A hospital database for location-based recommendations.

## 2. How it is made
The system is built as modular Python packages with a clear separation of concerns:

- Data and preprocessing (src/preprocess.py, src/data_prep.py)
  - Load data from CSV, clean missing values.
  - Encode categorical features and scale numerical features.
  - Optionally perform feature selection and handle class imbalance (e.g., SMOTE).

- Model training and evaluation (src/train.py)
  - Train multiple models such as Random Forest, XGBoost, CatBoost, and others.
  - Use cross-validation and grid search for hyperparameter tuning.
  - Compute metrics like accuracy, precision, recall, F1, and AUC.
  - Save the best model to disk for inference.

- Fairness analysis (src/fairness.py)
  - Evaluate bias across sensitive attributes such as age and sex.
  - Report fairness metrics such as demographic parity and equalized odds.

- Recommendations and hospital database (src/recommendations.py, src/database.py)
  - Convert risk probability to a risk level (low, moderate, high, critical).
  - Generate lifestyle advice and specialist guidance based on risk.
  - Query a SQLite hospital database for nearby care options.

- Web application (app/main.py)
  - Streamlit-based UI with patient input forms.
  - Real-time prediction and explanation display.
  - Batch CSV upload for multiple patients.
  - Export or report generation features.

- Supporting assets
  - docs/ for architecture, API, and user guides.
  - tests/ for unit and integration tests.
  - data/ for datasets and hospital data.
  - models/ for trained model artifacts.

## 3. How it works (end-to-end flow)
1. User inputs patient data through the Streamlit UI (age, sex, BP, cholesterol, etc.).
2. Inputs are mapped into the model feature format (including encoded categorical features).
3. The app loads a trained model from disk (models/best_heart_model.pkl).
4. The model predicts the probability of heart disease and a binary classification.
5. The system maps the probability into a risk level (low, moderate, high, critical).
6. Recommendations are generated based on the risk level and patient attributes.
7. If location details are provided, the app queries nearby hospitals for suggestions.
8. Results are visualized in charts and summary panels, and reports can be generated.

## 4. Top questions and answers a professor may ask

1) Q: What problem does the project solve?
   A: It predicts heart disease risk from clinical data, providing decision support with explainability and recommendations.

2) Q: What dataset is used and what are its features?
   A: It uses the Heart Failure Prediction dataset from Kaggle. Features include age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG, max HR, exercise angina, oldpeak, and ST slope. The target is HeartDisease (0/1).

3) Q: How do you handle categorical features?
   A: Categorical fields are encoded (label or one-hot style) during preprocessing to make them numeric for model input.

4) Q: How do you handle class imbalance?
   A: The pipeline can apply SMOTE on the training data to balance positive and negative classes.

5) Q: Which models were tried and how was the best model selected?
   A: Multiple models such as Random Forest, XGBoost, CatBoost, and others are trained. The best model is selected using metrics like F1 and ROC-AUC with cross-validation.

6) Q: What evaluation metrics are reported and why?
   A: Accuracy, precision, recall, F1, and AUC. This balances overall correctness (accuracy) and sensitivity to positive cases (recall), which is important for healthcare risk prediction.

7) Q: How do you interpret the model predictions?
   A: The system uses feature importance and SHAP-based explanations to show which features most influenced a prediction.

8) Q: What does the fairness module do?
   A: It evaluates bias across sensitive attributes (age, sex) using metrics such as demographic parity and equalized odds.

9) Q: What is the architecture of the system?
   A: A modular design with preprocessing, ML training, fairness, and recommendation components, plus a Streamlit UI and a SQLite hospital database.

10) Q: How is the model deployed for users?
    A: It is packaged in a Streamlit app that loads the trained model and runs locally or can be deployed to a hosting service like Streamlit Cloud.

11) Q: How does batch processing work?
    A: Users upload a CSV file. The app preprocesses it, runs predictions for each row, and returns a results table.

12) Q: What are the main limitations?
    A: The dataset size and scope are limited, so generalization to different populations may be constrained. It is a decision-support tool, not a clinical diagnosis.

13) Q: How is data privacy considered?
    A: The system avoids storing personally identifiable data and focuses on aggregated or anonymized input features.

14) Q: What tests exist to validate the system?
    A: The repository includes unit and integration tests in the tests/ folder, along with pytest configuration.

15) Q: What would you improve in the next version?
    A: Increase dataset diversity, add external validation, improve calibration, and integrate continuous monitoring for model drift.
