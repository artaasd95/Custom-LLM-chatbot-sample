#!/usr/bin/env python3
"""Test runner script for the Custom LLM Chatbot project."""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n{description}")
        print("=" * len(description))
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def run_all_tests(args):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    if args.verbose:
        cmd.append("-vv")
    
    if args.stop_on_first_fail:
        cmd.append("-x")
    
    return run_command(cmd, "Running all tests")


def run_unit_tests(args):
    """Run unit tests only."""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "not integration and not slow"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    return run_command(cmd, "Running unit tests")


def run_integration_tests(args):
    """Run integration tests only."""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "integration"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Running integration tests")


def run_specific_test(args):
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", args.test_path, "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if args.verbose:
        cmd.append("-vv")
    
    return run_command(cmd, f"Running specific test: {args.test_path}")


def run_tests_by_module(args):
    """Run tests for a specific module."""
    module_test_map = {
        "config": "tests/test_config.py",
        "data": "tests/test_data_processor.py",
        "training": "tests/test_training.py",
        "serving": "tests/test_serving.py",
        "monitoring": "tests/test_monitoring.py"
    }
    
    if args.module not in module_test_map:
        print(f"Unknown module: {args.module}")
        print(f"Available modules: {', '.join(module_test_map.keys())}")
        return False
    
    test_file = module_test_map[args.module]
    cmd = ["python", "-m", "pytest", test_file, "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, f"Running tests for module: {args.module}")


def run_performance_tests(args):
    """Run performance/benchmark tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "slow", "--benchmark-only"]
    
    return run_command(cmd, "Running performance tests")


def run_linting(args):
    """Run code linting."""
    success = True
    
    # Run flake8
    if run_command(["python", "-m", "flake8", "src/", "tests/"], "Running flake8 linting"):
        print("✓ flake8 passed")
    else:
        print("✗ flake8 failed")
        success = False
    
    # Run black check
    if run_command(["python", "-m", "black", "--check", "src/", "tests/"], "Checking code formatting with black"):
        print("✓ black formatting check passed")
    else:
        print("✗ black formatting check failed")
        success = False
    
    # Run isort check
    if run_command(["python", "-m", "isort", "--check-only", "src/", "tests/"], "Checking import sorting with isort"):
        print("✓ isort check passed")
    else:
        print("✗ isort check failed")
        success = False
    
    return success


def run_type_checking(args):
    """Run type checking with mypy."""
    return run_command(["python", "-m", "mypy", "src/"], "Running type checking with mypy")


def run_security_check(args):
    """Run security checks."""
    success = True
    
    # Run bandit security check
    if run_command(["python", "-m", "bandit", "-r", "src/"], "Running security check with bandit"):
        print("✓ bandit security check passed")
    else:
        print("✗ bandit security check failed")
        success = False
    
    # Run safety check for dependencies
    if run_command(["python", "-m", "safety", "check"], "Checking dependencies for security vulnerabilities"):
        print("✓ safety check passed")
    else:
        print("✗ safety check failed")
        success = False
    
    return success


def generate_coverage_report(args):
    """Generate detailed coverage report."""
    success = True
    
    # Run tests with coverage
    cmd = ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html", "--cov-report=xml", "--cov-report=term"]
    
    if run_command(cmd, "Generating coverage report"):
        print("\n✓ Coverage report generated")
        print("HTML report: htmlcov/index.html")
        print("XML report: coverage.xml")
    else:
        print("✗ Failed to generate coverage report")
        success = False
    
    return success


def run_full_ci_pipeline(args):
    """Run the full CI pipeline (linting, type checking, tests, coverage)."""
    print("Running full CI pipeline...")
    print("=" * 50)
    
    success = True
    
    # Step 1: Linting
    if not run_linting(args):
        print("\n❌ Linting failed")
        success = False
    else:
        print("\n✅ Linting passed")
    
    # Step 2: Type checking
    if not run_type_checking(args):
        print("\n❌ Type checking failed")
        success = False
    else:
        print("\n✅ Type checking passed")
    
    # Step 3: Security checks
    if not run_security_check(args):
        print("\n❌ Security checks failed")
        success = False
    else:
        print("\n✅ Security checks passed")
    
    # Step 4: Unit tests
    if not run_unit_tests(args):
        print("\n❌ Unit tests failed")
        success = False
    else:
        print("\n✅ Unit tests passed")
    
    # Step 5: Integration tests
    if not run_integration_tests(args):
        print("\n❌ Integration tests failed")
        success = False
    else:
        print("\n✅ Integration tests passed")
    
    # Step 6: Coverage report
    if not generate_coverage_report(args):
        print("\n❌ Coverage report generation failed")
        success = False
    else:
        print("\n✅ Coverage report generated")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All CI pipeline steps passed!")
    else:
        print("💥 CI pipeline failed. Please fix the issues above.")
    
    return success


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test runner for Custom LLM Chatbot")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # All tests
    all_parser = subparsers.add_parser("all", help="Run all tests")
    all_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    all_parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    all_parser.add_argument("--markers", help="Run tests with specific markers")
    all_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    all_parser.add_argument("--stop-on-first-fail", action="store_true", help="Stop on first failure")
    
    # Unit tests
    unit_parser = subparsers.add_parser("unit", help="Run unit tests only")
    unit_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    unit_parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    
    # Integration tests
    integration_parser = subparsers.add_parser("integration", help="Run integration tests only")
    integration_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    # Specific test
    specific_parser = subparsers.add_parser("specific", help="Run a specific test")
    specific_parser.add_argument("test_path", help="Path to test file or test function")
    specific_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    specific_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Module tests
    module_parser = subparsers.add_parser("module", help="Run tests for a specific module")
    module_parser.add_argument("module", choices=["config", "data", "training", "serving", "monitoring"],
                              help="Module to test")
    module_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    # Performance tests
    perf_parser = subparsers.add_parser("performance", help="Run performance tests")
    
    # Linting
    lint_parser = subparsers.add_parser("lint", help="Run code linting")
    
    # Type checking
    type_parser = subparsers.add_parser("typecheck", help="Run type checking")
    
    # Security check
    security_parser = subparsers.add_parser("security", help="Run security checks")
    
    # Coverage report
    coverage_parser = subparsers.add_parser("coverage", help="Generate coverage report")
    
    # Full CI pipeline
    ci_parser = subparsers.add_parser("ci", help="Run full CI pipeline")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run the appropriate command
    command_map = {
        "all": run_all_tests,
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "specific": run_specific_test,
        "module": run_tests_by_module,
        "performance": run_performance_tests,
        "lint": run_linting,
        "typecheck": run_type_checking,
        "security": run_security_check,
        "coverage": generate_coverage_report,
        "ci": run_full_ci_pipeline
    }
    
    success = command_map[args.command](args)
    
    if success:
        print("\n✅ Command completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Command failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()