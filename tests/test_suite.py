"""
Comprehensive test suite for Heart Failure Prediction System.
"""

import pytest
import sys
import os
import coverage

# Test configuration
COVERAGE_THRESHOLD = 80  # Minimum coverage percentage
TEST_TIMEOUT = 300  # Test timeout in seconds

def run_all_tests():
    """Run all tests with coverage reporting."""
    print("=== Heart Failure Prediction System - Test Suite ===")
    print("Running comprehensive tests...\n")
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    try:
        # Run pytest with comprehensive options
        exit_code = pytest.main([
            '-v',                    # Verbose output
            '--tb=short',           # Short traceback format
            '--strict-markers',     # Strict marker enforcement
            '--durations=10',       # Show 10 slowest tests
            '--cov=src',           # Coverage for src directory
            '--cov=app',           # Coverage for app directory
            '--cov-report=html',    # HTML coverage report
            '--cov-report=term-missing',  # Terminal coverage with missing lines
            f'--cov-fail-under={COVERAGE_THRESHOLD}',  # Fail if coverage below threshold
            '--junit-xml=test_results.xml',  # JUnit XML report
            'tests/',              # Test directory
            '-x',                  # Stop on first failure (optional)
        ])
        
        return exit_code
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1
    finally:
        cov.stop()
        cov.save()

def run_specific_test_category(category):
    """Run tests for a specific category."""
    valid_categories = ['unit', 'integration', 'model', 'database', 'streamlit', 'performance', 'edge_cases']
    
    if category not in valid_categories:
        print(f"Invalid category: {category}")
        print(f"Valid categories: {', '.join(valid_categories)}")
        return 1
    
    print(f"Running {category} tests...")
    
    exit_code = pytest.main([
        '-v',
        '-m', category,
        '--tb=short',
        'tests/'
    ])
    
    return exit_code

def run_quick_tests():
    """Run only quick unit tests for development."""
    print("Running quick unit tests...")
    
    exit_code = pytest.main([
        '-v',
        '-m', 'unit and not slow',
        '--tb=short',
        'tests/'
    ])
    
    return exit_code

def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    
    exit_code = pytest.main([
        '-v',
        '-m', 'integration',
        '--tb=short',
        'tests/'
    ])
    
    return exit_code

def run_performance_tests():
    """Run performance tests."""
    print("Running performance tests...")
    
    exit_code = pytest.main([
        '-v',
        '-m', 'performance',
        '--tb=short',
        'tests/'
    ])
    
    return exit_code

def generate_test_report():
    """Generate comprehensive test report."""
    print("\n=== Test Report Generation ===")
    
    # Run tests with detailed reporting
    exit_code = pytest.main([
        '--tb=long',
        '--cov=src',
        '--cov=app', 
        '--cov-report=html:htmlcov',
        '--cov-report=xml:coverage.xml',
        '--cov-report=term',
        '--junit-xml=test_results.xml',
        '--html=test_report.html',
        '--self-contained-html',
        'tests/'
    ])
    
    print("\nGenerated reports:")
    print("- HTML Coverage Report: htmlcov/index.html")
    print("- XML Coverage Report: coverage.xml") 
    print("- JUnit XML Report: test_results.xml")
    print("- HTML Test Report: test_report.html")
    
    return exit_code

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Heart Failure Prediction Test Suite')
    parser.add_argument('--category', '-c', 
                       choices=['unit', 'integration', 'model', 'database', 'streamlit', 'performance', 'edge_cases'],
                       help='Run tests for specific category')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run only quick unit tests')
    parser.add_argument('--integration', '-i', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--performance', '-p', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate comprehensive test report')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all tests with coverage')
    
    args = parser.parse_args()
    
    if args.category:
        exit_code = run_specific_test_category(args.category)
    elif args.quick:
        exit_code = run_quick_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    elif args.performance:
        exit_code = run_performance_tests()
    elif args.report:
        exit_code = generate_test_report()
    elif args.all:
        exit_code = run_all_tests()
    else:
        # Default: run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
