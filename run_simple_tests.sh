#!/bin/bash
# Simple test runner for working tests

set -e

echo "======================================"
echo "Running Simplified Test Suite"
echo "======================================"
echo ""

# Run tests
pytest tests_simple/ -v --tb=short

echo ""
echo "======================================"
echo "Tests Complete!"
echo "======================================"
