"""
Enhanced preprocessing pipeline for heart failure prediction.
Includes SMOTE for class imbalance, feature selection, proper encoding, and scaling.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePreprocessor:
    """
    Comprehensive preprocessor for heart disease prediction data.
    """
    
    def __init__(self, data_path=None, target_column='HeartDisease'):
        self.data_path = data_path
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
        
    def load_data(self):
        """Load the heart disease dataset."""
        if self.data_path is None:
            # Default to combined dataset
            self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_heart.csv')
        
        df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # For numeric columns, use median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, use mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} missing values with mode: {mode_val}")
        
        # Handle Cholesterol = 0 (common data quality issue)
        if 'Cholesterol' in df.columns:
            zero_cholesterol = (df['Cholesterol'] == 0).sum()
            if zero_cholesterol > 0:
                median_cholesterol = df[df['Cholesterol'] > 0]['Cholesterol'].median()
                df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = median_cholesterol
                print(f"Replaced {zero_cholesterol} zero cholesterol values with median: {median_cholesterol}")
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using appropriate encoding methods."""
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != self.target_column]
        
        encoded_df = df.copy()
        
        for col in categorical_columns:
            if col in ['Sex', 'ExerciseAngina', 'FastingBS']:
                # Binary categorical features - use label encoding
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    encoded_df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    encoded_df[col] = self.label_encoders[col].transform(df[col])
                    
            elif col in ['ChestPainType', 'RestingECG', 'ST_Slope']:
                # Multi-class categorical features - use one-hot encoding
                if fit:
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    # Store column names for transform
                    setattr(self, f'{col}_columns', dummies.columns.tolist())
                else:
                    # Create dummy variables for transform
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    # Ensure same columns as training
                    expected_cols = getattr(self, f'{col}_columns')
                    for expected_col in expected_cols:
                        if expected_col not in dummies.columns:
                            dummies[expected_col] = 0
                    dummies = dummies[expected_cols]  # Reorder columns
                
                # Add dummy columns to dataframe
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
                # Remove original column
                encoded_df.drop(col, axis=1, inplace=True)
        
        return encoded_df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features."""
        # Identify numeric columns (excluding target)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != self.target_column]
        
        scaled_df = df.copy()
        
        if fit:
            scaled_df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        else:
            scaled_df[numeric_columns] = self.scaler.transform(df[numeric_columns])
            
        return scaled_df
    
    def select_features(self, X, y, k=10):
        """Select top k features using statistical tests."""
        # Determine the test based on feature types
        # For mixed features, use f_classif
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        feature_names = X.columns
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = feature_names[selected_indices].tolist()
        
        print(f"Selected features: {self.selected_features}")
        print(f"Feature scores: {self.feature_selector.scores_[selected_indices]}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def apply_smote(self, X, y, random_state=42):
        """Apply SMOTE for handling class imbalance."""
        print(f"Original class distribution: {pd.Series(y).value_counts().sort_index()}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE class distribution: {pd.Series(y_resampled).value_counts().sort_index()}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def fit_transform(self, df):
        """Fit the preprocessor and transform the data."""
        print("=== Fitting Preprocessor ===")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        X_encoded = self.encode_categorical_features(X, fit=True)
        
        # Store training columns for transform alignment
        self.training_columns = X_encoded.columns.tolist()
        
        # Scale features
        print("\nScaling features...")
        X_scaled = self.scale_features(X_encoded, fit=True)
        
        # Feature selection
        print("\nSelecting features...")
        X_selected = self.select_features(X_scaled, y, k=min(10, X_scaled.shape[1]))
        
        self.is_fitted = True
        print("\nPreprocessor fitted successfully!")
        
        return X_selected, y
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        print("=== Transforming Data ===")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Separate features and target (if present)
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df.copy()
            y = None
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X, fit=False)
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, fit=False)
        
        # Select features
        if self.selected_features:
            X_selected = X_scaled[self.selected_features]
        else:
            X_selected = X_scaled
        
        return X_selected, y
    
    def save(self, filepath):
        """Save the fitted preprocessor."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a fitted preprocessor."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Train class distribution: {pd.Series(y_train).value_counts().sort_index()}")
    print(f"Test class distribution: {pd.Series(y_test).value_counts().sort_index()}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main preprocessing pipeline."""
    print("=== Heart Disease Data Preprocessing ===\n")
    
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor()
    
    # Load data
    df = preprocessor.load_data()
    
    # Fit and transform data
    X, y = preprocessor.fit_transform(df)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Apply SMOTE to training data only
    print("\n=== Applying SMOTE to Training Data ===")
    X_train_balanced, y_train_balanced = preprocessor.apply_smote(X_train, y_train)
    
    # Save preprocessor
    preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    
    # Save processed datasets
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Save training data (balanced)
    pd.concat([X_train_balanced, y_train_balanced], axis=1).to_csv(
        os.path.join(data_dir, 'train_balanced.csv'), index=False
    )
    
    # Save training data (original)
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(data_dir, 'train_original.csv'), index=False
    )
    
    # Save test data
    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(data_dir, 'test.csv'), index=False
    )
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Balanced training set: {X_train_balanced.shape}")
    print(f"Original training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Selected features: {len(X.columns)}")
    
    return preprocessor, X_train_balanced, y_train_balanced, X_test, y_test

if __name__ == "__main__":
    preprocessor, X_train, y_train, X_test, y_test = main()