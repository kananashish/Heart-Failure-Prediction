#!/usr/bin/env python
"""
Comprehensive import verification for Streamlit Cloud deployment.
Tests all critical dependencies used by the app.
"""

import sys

def test_imports():
    """Test all required imports"""
    errors = []
    
    print("Testing critical imports for Streamlit Cloud deployment...\n")
    
    imports_to_test = [
        ("streamlit", "streamlit"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("plotly.express", "px"),
        ("plotly.graph_objects", "go"),
        ("plotly.subplots", "make_subplots"),
        ("joblib", "joblib"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("catboost", "catboost"),
        ("shap", "shap"),
        ("fairlearn", "fairlearn"),
        ("imblearn", "imbalanced-learn"),
        ("sqlite3", "sqlite3"),
    ]
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name:<25} OK")
        except ImportError as e:
            error_msg = f"✗ {display_name:<25} FAILED: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    print("\n" + "="*60)
    if errors:
        print(f"\n❌ {len(errors)} import(s) failed:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("\n✅ All critical imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
