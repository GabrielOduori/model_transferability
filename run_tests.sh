#!/bin/bash
# Test runner script for model_transferability project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}Model Transferability Test Suite${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""

# Parse command line arguments
TEST_TYPE=${1:-all}
VERBOSE=${2:-}

# Function to run tests
run_tests() {
    local test_path=$1
    local description=$2

    echo -e "${YELLOW}Running $description...${NC}"

    if [ "$VERBOSE" = "-v" ] || [ "$VERBOSE" = "--verbose" ]; then
        pytest "$test_path" -v --tb=short
    else
        pytest "$test_path" --tb=short
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $description passed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        exit 1
    fi
    echo ""
}

# Main test execution
case $TEST_TYPE in
    all)
        echo "Running all tests..."
        run_tests "tests/" "All Tests"
        ;;

    unit)
        echo "Running unit tests only..."
        run_tests "tests/test_models tests/test_transfer_methods tests/test_evaluation" "Unit Tests"
        ;;

    integration)
        echo "Running integration tests only..."
        run_tests "tests/test_integration" "Integration Tests"
        ;;

    models)
        echo "Running model tests..."
        run_tests "tests/test_models" "Model Tests"
        ;;

    transfer)
        echo "Running transfer method tests..."
        run_tests "tests/test_transfer_methods" "Transfer Method Tests"
        ;;

    metrics)
        echo "Running metrics tests..."
        run_tests "tests/test_evaluation" "Metrics Tests"
        ;;

    coverage)
        echo "Running tests with coverage report..."
        pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
        echo ""
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;

    quick)
        echo "Running quick tests (excluding slow tests)..."
        pytest tests/ -m "not slow" --tb=short
        ;;

    help)
        echo "Usage: ./run_tests.sh [TEST_TYPE] [OPTIONS]"
        echo ""
        echo "TEST_TYPE options:"
        echo "  all          - Run all tests (default)"
        echo "  unit         - Run unit tests only"
        echo "  integration  - Run integration tests only"
        echo "  models       - Run model tests only"
        echo "  transfer     - Run transfer method tests only"
        echo "  metrics      - Run metrics tests only"
        echo "  coverage     - Run with coverage report"
        echo "  quick        - Run quick tests (exclude slow tests)"
        echo "  help         - Show this help message"
        echo ""
        echo "OPTIONS:"
        echo "  -v, --verbose  - Verbose output"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh all -v"
        echo "  ./run_tests.sh unit"
        echo "  ./run_tests.sh coverage"
        ;;

    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}All requested tests completed!${NC}"
echo -e "${GREEN}===================================${NC}"
