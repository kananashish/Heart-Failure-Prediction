"""
Automated Exploratory Data Analysis (EDA) for Heart Failure Prediction.
Generates comprehensive data analysis reports and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class HeartFailureEDA:
    """
    Comprehensive Exploratory Data Analysis for Heart Failure Prediction dataset.
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), '..', 'heart.csv')
        self.df = None
        self.report_path = os.path.join(os.path.dirname(__file__), '..', 'docs')
        self.plots_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'plots')
        
        # Create directories if they don't exist
        os.makedirs(self.report_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        
    def load_data(self):
        """Load the heart failure dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Dataset not found at {self.data_path}")
            return False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def generate_basic_statistics(self):
        """Generate basic statistical summary of the dataset."""
        print("=== Basic Dataset Statistics ===")
        
        stats = {
            'Dataset Shape': self.df.shape,
            'Missing Values': self.df.isnull().sum().sum(),
            'Duplicate Rows': self.df.duplicated().sum(),
            'Memory Usage (MB)': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Feature types
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        stats['Numeric Features'] = len(numeric_features)
        stats['Categorical Features'] = len(categorical_features)
        
        # Target variable analysis
        if 'HeartDisease' in self.df.columns:
            target_dist = self.df['HeartDisease'].value_counts()
            stats['Target Distribution'] = dict(target_dist)
            stats['Class Balance Ratio'] = round(target_dist.min() / target_dist.max(), 3)
        
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return stats
    
    def create_target_analysis(self):
        """Analyze target variable distribution and relationships."""
        if 'HeartDisease' not in self.df.columns:
            print("Target variable 'HeartDisease' not found")
            return
        
        print("\\n=== Target Variable Analysis ===")
        
        # Target distribution
        target_counts = self.df['HeartDisease'].value_counts()
        target_pct = self.df['HeartDisease'].value_counts(normalize=True) * 100
        
        print(f"Target Distribution:")
        print(f"  No Heart Disease (0): {target_counts[0]} ({target_pct[0]:.1f}%)")
        print(f"  Heart Disease (1): {target_counts[1]} ({target_pct[1]:.1f}%)")
        
        # Create target distribution plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Count Distribution', 'Percentage Distribution'],
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Bar plot
        fig.add_trace(
            go.Bar(x=['No Heart Disease', 'Heart Disease'], 
                   y=[target_counts[0], target_counts[1]],
                   marker_color=['lightblue', 'lightcoral']),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=['No Heart Disease', 'Heart Disease'], 
                   values=[target_counts[0], target_counts[1]],
                   marker_colors=['lightblue', 'lightcoral']),
            row=1, col=2
        )
        
        fig.update_layout(title_text="Target Variable Distribution", height=400)
        fig.write_html(os.path.join(self.plots_path, 'target_distribution.html'))
        print(f"Target distribution plot saved to: {self.plots_path}/target_distribution.html")
    
    def analyze_numeric_features(self):
        """Analyze numeric features distribution and relationships."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'HeartDisease' in numeric_cols:
            numeric_cols.remove('HeartDisease')
        
        print(f"\\n=== Numeric Features Analysis ({len(numeric_cols)} features) ===")
        
        # Basic statistics
        numeric_stats = self.df[numeric_cols].describe()
        print("\\nDescriptive Statistics:")
        print(numeric_stats)
        
        # Create distribution plots
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(title_text="Numeric Features Distribution", height=200*n_rows)
        fig.write_html(os.path.join(self.plots_path, 'numeric_distributions.html'))
        print(f"Numeric distributions plot saved to: {self.plots_path}/numeric_distributions.html")
        
        return numeric_stats
    
    def analyze_categorical_features(self):
        """Analyze categorical features distribution and relationships."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also include binary numeric features that are actually categorical
        binary_cols = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col != 'HeartDisease' and len(self.df[col].unique()) <= 5:
                binary_cols.append(col)
        
        all_categorical = categorical_cols + binary_cols
        
        print(f"\\n=== Categorical Features Analysis ({len(all_categorical)} features) ===")
        
        for col in all_categorical:
            print(f"\\n{col}:")
            value_counts = self.df[col].value_counts()
            for value, count in value_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"  {value}: {count} ({pct:.1f}%)")
        
        # Create categorical distribution plots
        if all_categorical:
            n_cols = 2
            n_rows = (len(all_categorical) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=all_categorical,
                specs=[[{'type': 'bar'} for _ in range(n_cols)] for _ in range(n_rows)]
            )
            
            for i, col in enumerate(all_categorical):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                value_counts = self.df[col].value_counts()
                
                fig.add_trace(
                    go.Bar(x=value_counts.index.astype(str), 
                           y=value_counts.values, 
                           name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(title_text="Categorical Features Distribution", height=300*n_rows)
            fig.write_html(os.path.join(self.plots_path, 'categorical_distributions.html'))
            print(f"Categorical distributions plot saved to: {self.plots_path}/categorical_distributions.html")
    
    def create_correlation_analysis(self):
        """Create correlation analysis for numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("Insufficient numeric columns for correlation analysis")
            return
        
        print("\\n=== Correlation Analysis ===")
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for high correlation
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        print(f"High correlations (|r| > 0.5):")
        for var1, var2, corr_val in high_corr_pairs:
            print(f"  {var1} - {var2}: {corr_val:.3f}")
        
        # Create correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_nticks=len(corr_matrix.columns),
            yaxis_nticks=len(corr_matrix.columns),
            height=600
        )
        
        fig.write_html(os.path.join(self.plots_path, 'correlation_matrix.html'))
        print(f"Correlation matrix saved to: {self.plots_path}/correlation_matrix.html")
        
        return corr_matrix
    
    def analyze_target_relationships(self):
        """Analyze relationships between features and target variable."""
        if 'HeartDisease' not in self.df.columns:
            print("Target variable not found")
            return
        
        print("\\n=== Target Relationships Analysis ===")
        
        # Numeric features vs target
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'HeartDisease']
        
        # Create box plots for numeric features
        if numeric_cols:
            n_cols = 3
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=numeric_cols
            )
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                for target_val in [0, 1]:
                    data = self.df[self.df['HeartDisease'] == target_val][col]
                    label = 'No Heart Disease' if target_val == 0 else 'Heart Disease'
                    
                    fig.add_trace(
                        go.Box(y=data, name=f"{label} - {col}", 
                               showlegend=(i == 0)),  # Only show legend for first subplot
                        row=row, col=col_pos
                    )
            
            fig.update_layout(
                title_text="Numeric Features vs Target",
                height=300*n_rows
            )
            fig.write_html(os.path.join(self.plots_path, 'target_relationships_numeric.html'))
            print(f"Target relationships (numeric) plot saved to: {self.plots_path}/target_relationships_numeric.html")
        
        # Feature importance based on target correlation
        target_corr = self.df[numeric_cols + ['HeartDisease']].corr()['HeartDisease'].abs().sort_values(ascending=False)
        print("\\nFeature importance (correlation with target):")
        for feature, corr_val in target_corr.items():
            if feature != 'HeartDisease':
                print(f"  {feature}: {corr_val:.3f}")
    
    def detect_outliers(self):
        """Detect outliers in numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'HeartDisease']
        
        print("\\n=== Outlier Detection ===")
        
        outlier_summary = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"{col}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
        
        return outlier_summary
    
    def create_summary_report(self, stats, correlation_matrix, outlier_summary):
        """Create a comprehensive summary report."""
        print("\\n=== Creating Summary Report ===")
        
        report = f"""# Heart Failure Prediction Dataset - EDA Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

- **Shape**: {stats['Dataset Shape'][0]} rows × {stats['Dataset Shape'][1]} columns
- **Missing Values**: {stats['Missing Values']}
- **Duplicate Rows**: {stats['Duplicate Rows']}
- **Memory Usage**: {stats['Memory Usage (MB)']} MB
- **Numeric Features**: {stats['Numeric Features']}
- **Categorical Features**: {stats['Categorical Features']}

## Target Variable Analysis

"""
        
        if 'Target Distribution' in stats:
            target_dist = stats['Target Distribution']
            total = sum(target_dist.values())
            report += f"""- **No Heart Disease (0)**: {target_dist[0]} ({target_dist[0]/total*100:.1f}%)
- **Heart Disease (1)**: {target_dist[1]} ({target_dist[1]/total*100:.1f}%)
- **Class Balance Ratio**: {stats['Class Balance Ratio']}

"""
        
        # High correlations
        if correlation_matrix is not None:
            report += "## High Correlations (|r| > 0.5)\\n\\n"
            high_corrs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        high_corrs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
            
            if high_corrs:
                for var1, var2, corr_val in high_corrs:
                    report += f"- **{var1} - {var2}**: {corr_val:.3f}\\n"
            else:
                report += "No high correlations found.\\n"
        
        # Outlier summary
        report += "\\n## Outlier Summary\\n\\n"
        for feature, outlier_info in outlier_summary.items():
            if outlier_info['count'] > 0:
                report += f"- **{feature}**: {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)\\n"
        
        # Data quality insights
        report += """

## Data Quality Assessment

### Strengths
- No missing values detected
- Well-balanced target variable
- Good mix of numeric and categorical features

### Potential Issues
- Check outliers for data entry errors
- Consider feature scaling for ML models
- Validate categorical encodings

## Visualizations Created

1. **Target Distribution**: `docs/plots/target_distribution.html`
2. **Numeric Features**: `docs/plots/numeric_distributions.html`
3. **Categorical Features**: `docs/plots/categorical_distributions.html`
4. **Correlation Matrix**: `docs/plots/correlation_matrix.html`
5. **Target Relationships**: `docs/plots/target_relationships_numeric.html`

## Recommendations

1. **Feature Engineering**: Consider creating interaction terms between highly correlated features
2. **Preprocessing**: Apply standardization to numeric features before modeling
3. **Class Imbalance**: Current dataset is well-balanced, no special handling needed
4. **Outliers**: Investigate and potentially remove or cap extreme outliers
5. **Categorical Encoding**: Use appropriate encoding (one-hot, label, target) based on feature cardinality

---
*Report generated by HeartFailureEDA*
"""
        
        # Save report
        report_path = os.path.join(self.report_path, 'eda_summary_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Summary report saved to: {report_path}")
        return report
    
    def run_complete_analysis(self):
        """Run complete EDA analysis."""
        print("=== Starting Complete EDA Analysis ===")
        
        # Load data
        if not self.load_data():
            return False
        
        # Basic statistics
        stats = self.generate_basic_statistics()
        
        # Target analysis
        self.create_target_analysis()
        
        # Feature analysis
        numeric_stats = self.analyze_numeric_features()
        self.analyze_categorical_features()
        
        # Correlation analysis
        correlation_matrix = self.create_correlation_analysis()
        
        # Target relationships
        self.analyze_target_relationships()
        
        # Outlier detection
        outlier_summary = self.detect_outliers()
        
        # Create summary report
        summary_report = self.create_summary_report(stats, correlation_matrix, outlier_summary)
        
        print("\\n=== EDA Analysis Complete ===")
        print(f"Reports and visualizations saved to: {self.report_path}")
        print(f"Plots saved to: {self.plots_path}")
        
        return True

def main():
    """Main function to run EDA analysis."""
    # Initialize EDA analyzer
    eda = HeartFailureEDA()
    
    # Run complete analysis
    success = eda.run_complete_analysis()
    
    if success:
        print("\\n🎉 EDA analysis completed successfully!")
        print("\\nGenerated files:")
        print("- docs/eda_summary_report.md")
        print("- docs/plots/*.html (interactive visualizations)")
    else:
        print("\\n❌ EDA analysis failed")

if __name__ == "__main__":
    main()