"""
Advanced ML Training and Evaluation module for heart failure prediction.
Includes multiple models, cross-validation, SHAP explanations, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model training and evaluation class.
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different ML models with their parameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False, iterations=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f"Initialized {len(self.models)} models for training")
    
    def perform_cross_validation(self, X, y, cv_folds=5):
        """Perform cross-validation for all models."""
        print(f"\n=== Cross-Validation with {cv_folds} folds ===")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision', n_jobs=-1)
            cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall', n_jobs=-1)
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
            cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
            
            cv_results[name] = {
                'accuracy': cv_scores,
                'precision': cv_precision,
                'recall': cv_recall,
                'f1': cv_f1,
                'auc': cv_auc,
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'precision_mean': cv_precision.mean(),
                'recall_mean': cv_recall.mean(),
                'f1_mean': cv_f1.mean(),
                'auc_mean': cv_auc.mean()
            }
            
            print(f"  Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})") 
            print(f"  AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std() * 2:.4f})")
        
        return cv_results
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate on test set."""
        print(f"\n=== Training Models ===")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            self.results[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            if metrics['auc']:
                print(f"  AUC: {metrics['auc']:.4f}")
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost'):
        """Perform hyperparameter tuning for a specific model."""
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        self.trained_models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def explain_model_predictions(self, model_name, X_train, X_test, max_display=20):
        """Generate SHAP explanations for model predictions."""
        print(f"\n=== SHAP Explanation for {model_name} ===")
        
        if model_name not in self.trained_models:
            print(f"Model {model_name} not trained yet")
            return None
        
        model = self.trained_models[model_name]
        
        try:
            # Create SHAP explainer
            if model_name in ['XGBoost', 'Random Forest', 'Gradient Boosting']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100))
                shap_values = explainer.shap_values(X_test.sample(100))
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get positive class SHAP values
                X_test = X_test.sample(100)
            
            # Create plots directory
            plots_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{model_name}_shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{model_name}_shap_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP plots saved for {model_name}")
            return shap_values
            
        except Exception as e:
            print(f"Error generating SHAP explanations for {model_name}: {e}")
            return None
    
    def create_results_comparison(self):
        """Create comprehensive results comparison."""
        print(f"\n=== Model Performance Comparison ===")
        
        # Create results DataFrame
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC': metrics['auc'] if metrics['auc'] else 0
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(results_df.round(4))
        
        # Find best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name}")
        print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
        
        # Save results
        results_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_results.csv')
        results_df.to_csv(results_path, index=False)
        
        return results_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Model comparison plot
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'], 
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Bar plot comparison
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            results_df.set_index('Model')[metric].plot(kind='bar', ax=ax[row, col])
            ax[row, col].set_title(f'{metric} Comparison')
            ax[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrices for top 3 models
        top_models = results_df.nlargest(3, 'F1-Score')['Model'].values
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, model_name in enumerate(top_models):
            cm = self.results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved successfully")
    
    def save_best_model(self):
        """Save the best performing model."""
        if self.best_model is None:
            print("No best model identified yet")
            return
        
        model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'best_model.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'metrics': self.results[self.best_model_name]
            }, f)
        
        print(f"Best model ({self.best_model_name}) saved to {model_path}")

def load_processed_data():
    """Load preprocessed data."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Load training data (balanced)
    train_df = pd.read_csv(os.path.join(data_dir, 'train_balanced.csv'))
    X_train = train_df.drop('HeartDisease', axis=1)
    y_train = train_df['HeartDisease']
    
    # Load test data
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    X_test = test_df.drop('HeartDisease', axis=1)
    y_test = test_df['HeartDisease']
    
    print(f"Loaded training data: {X_train.shape}")
    print(f"Loaded test data: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def main():
    """Main training and evaluation pipeline."""
    print("=== Advanced ML Training and Evaluation ===\n")
    
    # Load processed data
    X_train, y_train, X_test, y_test = load_processed_data()
    
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Perform cross-validation
    cv_results = trainer.perform_cross_validation(X_train, y_train)
    
    # Train all models
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning for best models
    print("\n=== Hyperparameter Tuning ===")
    trainer.hyperparameter_tuning(X_train, y_train, 'XGBoost')
    trainer.hyperparameter_tuning(X_train, y_train, 'Random Forest')
    
    # Re-evaluate tuned models
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Create results comparison
    results_df = trainer.create_results_comparison()
    
    # Generate SHAP explanations for top 3 models
    top_models = results_df.head(3)['Model'].values
    for model_name in top_models:
        trainer.explain_model_predictions(model_name, X_train, X_test)
    
    # Create visualizations
    trainer.create_visualizations()
    
    # Save best model
    trainer.save_best_model()
    
    print(f"\n=== Training Complete ===")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Best F1-Score: {trainer.results[trainer.best_model_name]['f1']:.4f}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()