#!/usr/bin/env python
"""
Comprehensive Streamlit Cloud deployment test.
Tests all critical components before deployment.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

class DeploymentTester:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def test_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor >= 8:
            self.passed.append("Python version compatible (3.8+)")
            return True
        else:
            self.errors.append(f"Python 3.8+ required, got {version.major}.{version.minor}")
            return False
    
    def test_critical_imports(self):
        """Test all critical package imports"""
        imports = [
            "streamlit",
            "pandas",
            "numpy",
            "plotly.express",
            "plotly.graph_objects",
            "joblib",
            "sklearn",
            "xgboost",
            "catboost",
            "shap",
            "fairlearn",
            "imblearn",
        ]
        
        failed = []
        for module in imports:
            try:
                __import__(module)
            except ImportError as e:
                failed.append(f"{module}: {str(e)}")
        
        if failed:
            for error in failed:
                self.errors.append(f"Import failed: {error}")
            return False
        else:
            self.passed.append(f"All {len(imports)} critical imports successful")
            return True
    
    def test_requirements_file(self):
        """Verify requirements.txt is properly formatted"""
        req_file = Path("requirements.txt")
        if not req_file.exists():
            self.errors.append("requirements.txt not found")
            return False
        
        with open(req_file, 'r') as f:
            lines = f.readlines()
        
        # Check for common issues
        issues = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for invalid characters
            if line.count('=') > 2:
                issues.append(f"Line {i}: Invalid format '{line}'")
            
            # Check plotly version
            if line.startswith('plotly'):
                if '==' not in line:
                    issues.append(f"Line {i}: plotly should use pinned version (==), not {line}")
        
        if issues:
            for issue in issues:
                self.warnings.append(issue)
        
        self.passed.append("requirements.txt format valid")
        return True
    
    def test_app_file_exists(self):
        """Verify main app file exists"""
        app_file = Path("app/main.py")
        if not app_file.exists():
            self.errors.append("app/main.py not found")
            return False
        
        self.passed.append("app/main.py exists")
        return True
    
    def test_streamlit_config(self):
        """Verify Streamlit config exists"""
        config_file = Path(".streamlit/config.toml")
        if not config_file.exists():
            self.warnings.append(".streamlit/config.toml not found (optional but recommended)")
            return True
        
        self.passed.append(".streamlit/config.toml exists")
        return True
    
    def test_gitignore(self):
        """Check if sensitive files are in .gitignore"""
        gitignore_file = Path(".gitignore")
        if not gitignore_file.exists():
            self.warnings.append(".gitignore not found - ensure secrets are not committed")
            return True
        
        with open(gitignore_file, 'r') as f:
            content = f.read()
        
        critical_patterns = ['*.pkl', '*.joblib', '.env', 'secrets.toml']
        missing = [p for p in critical_patterns if p not in content]
        
        if missing:
            self.warnings.append(f"Consider adding to .gitignore: {', '.join(missing)}")
        
        self.passed.append(".gitignore configured")
        return True
    
    def test_src_module(self):
        """Test if src modules can be imported"""
        try:
            sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
            from preprocess import HeartDiseasePreprocessor
            self.passed.append("src.preprocess module loads successfully")
            return True
        except Exception as e:
            self.errors.append(f"Failed to import src.preprocess: {str(e)}")
            return False
    
    def test_model_files(self):
        """Check if model files exist and are accessible"""
        models_dir = Path("models")
        if not models_dir.exists():
            self.warnings.append("models/ directory not found - models must be generated")
            return True
        
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
        if model_files:
            self.passed.append(f"Found {len(model_files)} model file(s)")
        else:
            self.warnings.append("No pre-trained model files found - app must generate them on first run")
        
        return True
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "="*70)
        print("STREAMLIT CLOUD DEPLOYMENT VERIFICATION")
        print("="*70 + "\n")
        
        tests = [
            ("Python Version", self.test_python_version),
            ("Critical Imports", self.test_critical_imports),
            ("Requirements File", self.test_requirements_file),
            ("App File", self.test_app_file_exists),
            ("Streamlit Config", self.test_streamlit_config),
            (".gitignore", self.test_gitignore),
            ("Source Modules", self.test_src_module),
            ("Model Files", self.test_model_files),
        ]
        
        for test_name, test_func in tests:
            print(f"\n▶ {test_name}:")
            try:
                test_func()
            except Exception as e:
                self.errors.append(f"{test_name} test crashed: {str(e)}")
        
        self.print_report()
        return len(self.errors) == 0
    
    def print_report(self):
        """Print final report"""
        print("\n" + "="*70)
        print("DEPLOYMENT TEST RESULTS")
        print("="*70)
        
        if self.passed:
            print("\n✅ PASSED:")
            for item in self.passed:
                print(f"   ✓ {item}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for item in self.warnings:
                print(f"   ! {item}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for item in self.errors:
                print(f"   ✗ {item}")
            print("\n🚫 Deployment NOT READY - Fix errors above before deploying")
            return False
        else:
            print("\n✅ ✅ ✅  READY FOR DEPLOYMENT  ✅ ✅ ✅")
            print("\nNext steps:")
            print("  1. Commit all changes: git add . && git commit -m 'Deployment fixes'")
            print("  2. Push to GitHub: git push")
            print("  3. Redeploy on Streamlit Cloud (or deploy new app)")
            return True

if __name__ == "__main__":
    tester = DeploymentTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
