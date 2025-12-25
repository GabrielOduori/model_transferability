#!/bin/bash
# Quick fixes for easy test issues
# This will fix ~22 test failures automatically

echo "🔧 Applying quick test fixes..."
echo ""

# 1. Fix n_iter → num_iter (affects ~17 tests)
echo "📝 Fixing n_iter → num_iter..."
find tests/ -name "*.py" -type f -exec sed -i 's/n_iter=/num_iter=/g' {} \;
echo "   ✅ Fixed parameter naming in all test files"

# 2. Fix prior_variance → prior_variances (affects ~5 tests)
echo "📝 Fixing prior_variance → prior_variances..."
sed -i 's/prior_variance=/prior_variances=/g' tests/test_transfer_methods/test_prior_tempering.py
echo "   ✅ Fixed prior tempering parameter naming"

echo ""
echo "✅ Automatic fixes complete!"
echo ""
echo "Results:"
echo "  - Fixed ~22 test failures automatically"
echo "  - Expected improvement: 61% → ~80% pass rate"
echo ""
echo "Run tests to verify:"
echo "  pytest tests/ -v"
echo ""
echo "Note: Some manual fixes still needed for:"
echo "  - train_tempered_gp signature changes (~10 tests)"
echo "  - DPTR test assertions (~3 tests)"
echo "  - Integration test API updates (~9 tests)"
