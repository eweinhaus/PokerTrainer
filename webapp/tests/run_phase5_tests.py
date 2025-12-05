#!/usr/bin/env python3
"""
Test runner for Phase 5 tests
Runs all Phase 5 test suites and generates coverage report
"""

import unittest
import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    
    # Test modules to run
    test_modules = [
        'tests.coach.test_phase5_error_handling',
        'tests.integration.test_phase5_integration',
        'tests.e2e.test_phase5_e2e',
        'tests.performance.test_phase5_performance',
        'tests.edge_cases.test_phase5_edge_cases',
    ]
    
    # Load test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
    
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    
    if result.failures:
        for test, traceback in result.failures:
    
    if result.errors:
        for test, traceback in result.errors:
    
    return result.wasSuccessful()


def run_coverage_report():
    """Generate coverage report"""
    
    try:
        # Run coverage
        result = subprocess.run(
            ['coverage', 'run', '--source=coach,app', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_phase5*.py'],
            capture_output=True,
            text=True
        )
        
        # Generate report
        report_result = subprocess.run(
            ['coverage', 'report', '-m'],
            capture_output=True,
            text=True
        )
        
        
        # Generate HTML report
        html_result = subprocess.run(
            ['coverage', 'html'],
            capture_output=True,
            text=True
        )
        
        if html_result.returncode == 0:
        
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        return False


if __name__ == '__main__':
    # Run tests
    success = run_tests_with_coverage()
    
    # Try to generate coverage report
    if '--coverage' in sys.argv:
        run_coverage_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


