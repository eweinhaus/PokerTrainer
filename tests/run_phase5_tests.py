#!/usr/bin/env python3
"""
Test runner for Phase 5 tests
Runs all Phase 5 test suites and generates coverage report
"""

import unittest
import sys
import os
import subprocess

# Add webapp to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../webapp')))


def run_tests_with_coverage():
    """Run tests with coverage reporting"""
    
    # Test files to run
    test_files = [
        'tests/coach/test_phase5_error_handling.py',
        'tests/integration/test_phase5_integration.py',
        'tests/e2e/test_phase5_e2e.py',
        'tests/performance/test_phase5_performance.py',
        'tests/edge_cases/test_phase5_edge_cases.py',
    ]

    # Load test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_file in test_files:
        try:
            # Load module from file
            import importlib.util
            spec = importlib.util.spec_from_file_location(test_file.replace('/', '.').replace('.py', ''), test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite.addTests(loader.loadTestsFromModule(module))
        except Exception as e:
            print(f"Failed to load {test_file}: {e}")
            continue

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Results: {result.testsRun} tests run")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
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

        print("Coverage run completed")

        # Generate HTML report
        html_result = subprocess.run(
            ['coverage', 'html'],
            capture_output=True,
            text=True
        )
        
        if html_result.returncode == 0:
            print("HTML coverage report generated")

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


