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
    print("=" * 80)
    print("Phase 5 Test Suite")
    print("=" * 80)
    print()
    
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
            print(f"✓ Loaded {module_name}")
        except ImportError as e:
            print(f"✗ Failed to load {module_name}: {e}")
    
    print()
    print("=" * 80)
    print("Running Tests")
    print("=" * 80)
    print()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


def run_coverage_report():
    """Generate coverage report"""
    print()
    print("=" * 80)
    print("Generating Coverage Report")
    print("=" * 80)
    print()
    
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
        
        print(report_result.stdout)
        
        # Generate HTML report
        html_result = subprocess.run(
            ['coverage', 'html'],
            capture_output=True,
            text=True
        )
        
        if html_result.returncode == 0:
            print("\n✓ HTML coverage report generated in htmlcov/")
        
        return True
    except FileNotFoundError:
        print("⚠ Coverage tool not found. Install with: pip install coverage")
        return False
    except Exception as e:
        print(f"✗ Error generating coverage report: {e}")
        return False


if __name__ == '__main__':
    # Run tests
    success = run_tests_with_coverage()
    
    # Try to generate coverage report
    if '--coverage' in sys.argv:
        run_coverage_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


