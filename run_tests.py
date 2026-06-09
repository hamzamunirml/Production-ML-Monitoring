"""
Run all unit tests
"""

import unittest
import sys
import os

# Add tests directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """Run all unit tests"""
    print("="*60)
    print("RUNNING UNIT TESTS FOR BANK MARKETING SYSTEM")
    print("="*60)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        
        # Print failures
        if result.failures:
            print("\n📋 FAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        # Print errors
        if result.errors:
            print("\n📋 ERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run a specific test module"""
    print(f"\nRunning tests from: {test_module}")
    suite = unittest.defaultTestLoader.loadTestsFromName(f'tests.{test_module}')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unit tests')
    parser.add_argument('--test', type=str, help='Specific test to run (e.g., test_config)')
    args = parser.parse_args()
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)