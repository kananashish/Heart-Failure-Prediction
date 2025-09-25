"""
Data preparation and merging module for heart failure prediction project.
This module handles loading, cleaning, and combining multiple heart disease datasets.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_original_data():
    """Load the original heart failure prediction dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
    df = pd.read_csv(data_path)
    return df

def create_synthetic_uci_data(n_samples=500):
    """
    Create a synthetic UCI-style heart disease dataset for demonstration.
    In a real scenario, you would download the UCI heart disease dataset.
    """
    np.random.seed(42)
    
    # Generate synthetic data with similar distribution
    data = {
        'Age': np.random.normal(54, 12, n_samples).astype(int),
        'Sex': np.random.choice(['M', 'F'], n_samples, p=[0.68, 0.32]),
        'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples, 
                                        p=[0.25, 0.25, 0.35, 0.15]),
        'RestingBP': np.random.normal(132, 18, n_samples).astype(int),
        'Cholesterol': np.random.normal(198, 110, n_samples).astype(int),
        'FastingBS': np.random.choice([0, 1], n_samples, p=[0.77, 0.23]),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples, 
                                     p=[0.60, 0.25, 0.15]),
        'MaxHR': np.random.normal(136, 25, n_samples).astype(int),
        'ExerciseAngina': np.random.choice(['N', 'Y'], n_samples, p=[0.67, 0.33]),
        'Oldpeak': np.random.exponential(0.9, n_samples).round(1),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples, 
                                   p=[0.47, 0.48, 0.05]),
    }
    
    # Create synthetic target based on risk factors
    risk_score = (
        (data['Age'] > 60) * 0.3 +
        (np.array(data['Sex']) == 'M') * 0.2 +
        (np.array(data['ChestPainType']) == 'ASY') * 0.4 +
        (np.array(data['RestingBP']) > 140) * 0.2 +
        (np.array(data['Cholesterol']) > 240) * 0.2 +
        (np.array(data['FastingBS']) == 1) * 0.1 +
        (np.array(data['MaxHR']) < 120) * 0.3 +
        (np.array(data['ExerciseAngina']) == 'Y') * 0.4 +
        (np.array(data['Oldpeak']) > 1.0) * 0.3 +
        (np.array(data['ST_Slope']) == 'Flat') * 0.2
    )
    
    # Add some randomness and create binary target
    risk_score += np.random.normal(0, 0.2, n_samples)
    data['HeartDisease'] = (risk_score > 1.2).astype(int)
    
    # Ensure realistic ranges
    data['Age'] = np.clip(data['Age'], 28, 80)
    data['RestingBP'] = np.clip(data['RestingBP'], 90, 200)
    data['Cholesterol'] = np.clip(data['Cholesterol'], 0, 603)
    data['MaxHR'] = np.clip(data['MaxHR'], 60, 202)
    data['Oldpeak'] = np.clip(data['Oldpeak'], -2.6, 6.2)
    
    return pd.DataFrame(data)

def merge_datasets(df1, df2):
    """
    Merge two heart disease datasets with consistent column names.
    """
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")
    
    # Check if columns match
    if not set(df1.columns) == set(df2.columns):
        print("Warning: Column mismatch between datasets")
        print(f"DF1 columns: {set(df1.columns)}")
        print(f"DF2 columns: {set(df2.columns)}")
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Remove duplicates if any
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    removed_duplicates = initial_rows - len(combined_df)
    
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")
    
    print(f"Combined dataset shape: {combined_df.shape}")
    return combined_df

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    """
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # For Cholesterol, replace 0 values with median (common issue in heart datasets)
    if 'Cholesterol' in df.columns:
        cholesterol_median = df[df['Cholesterol'] > 0]['Cholesterol'].median()
        df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median
        print(f"Replaced 0 cholesterol values with median: {cholesterol_median}")
    
    # Handle any remaining missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

def create_feature_summary(df):
    """
    Create a summary of features in the dataset.
    """
    summary = {
        'total_samples': len(df),
        'target_distribution': df['HeartDisease'].value_counts().to_dict(),
        'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': list(df.select_dtypes(include=['object']).columns),
        'feature_stats': df.describe().to_dict()
    }
    return summary

def main():
    """
    Main function to prepare and merge datasets.
    """
    print("=== Heart Disease Data Preparation ===\n")
    
    # Load original dataset
    print("Loading original heart failure prediction dataset...")
    original_df = load_original_data()
    
    # Create synthetic UCI dataset (in real scenario, download from UCI repository)
    print("Creating synthetic UCI heart disease dataset...")
    uci_df = create_synthetic_uci_data()
    
    # Save synthetic UCI dataset
    uci_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'uci_heart.csv')
    uci_df.to_csv(uci_path, index=False)
    print(f"Synthetic UCI dataset saved to {uci_path}")
    
    # Merge datasets
    print("\nMerging datasets...")
    combined_df = merge_datasets(original_df, uci_df)
    
    # Handle missing values
    combined_df = handle_missing_values(combined_df)
    
    # Create feature summary
    summary = create_feature_summary(combined_df)
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Target distribution: {summary['target_distribution']}")
    print(f"Heart Disease rate: {summary['target_distribution'][1] / summary['total_samples']:.2%}")
    
    # Save combined dataset
    combined_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_heart.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined dataset saved to {combined_path}")
    
    return combined_df

if __name__ == "__main__":
    combined_data = main()