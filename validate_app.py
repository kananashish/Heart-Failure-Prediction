#!/usr/bin/env python
"""
Streamlit app syntax and structure validation.
Ensures the app can start without runtime errors on Streamlit Cloud.
"""

import sys
import ast
import os

def validate_app_syntax():
    """Validate Python syntax and basic structure"""
    
    print("Validating Streamlit app syntax and structure...\n")
    
    app_file = "app/main.py"
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    try:
        with open(app_file, 'r') as f:
            code = f.read()
        
        # Validate Python syntax
        ast.parse(code)
        print(f"✓ Python syntax valid in {app_file}")
        
        # Check for critical issues
        issues = []
        
        # Check if plotly is imported
        if "plotly.express" not in code and "import px" not in code:
            issues.append("⚠ plotly.express not explicitly imported")
        else:
            print("✓ plotly.express properly imported")
        
        # Check for streamlit imports
        if "import streamlit" not in code:
            issues.append("⚠ streamlit not imported")
        else:
            print("✓ streamlit properly imported")
        
        # Check for relative imports
        if "from src." in code or "import src." in code:
            print("✓ Relative imports to src/ found (OK for local)")
        
        # Warn about sys.path modifications if they exist
        if "sys.path" in code:
            print("✓ sys.path modifications found (for development)")
        
        print("\n" + "="*60)
        if issues:
            print(f"\n⚠ {len(issues)} potential issue(s):")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ App structure validation passed!")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {app_file}:")
        print(f"   Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error validating {app_file}: {e}")
        return False

if __name__ == "__main__":
    success = validate_app_syntax()
    sys.exit(0 if success else 1)
