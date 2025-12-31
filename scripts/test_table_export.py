"""
Test script to verify table export functionality.

This script creates a minimal results dictionary and tests the table export
function to ensure CSV and LaTeX tables are generated correctly.
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the export function from the experiment script
from experiments.run_model_transfer_experiments import export_results_tables


def create_test_results():
    """Create minimal test results matching the real structure."""
    return {
        'timestamp': '20241228_test',
        'source': 'Test Source',
        'target': 'Test Target',
        'fusiongp_prior_tempering': {
            'experiment': 'FusionGP_Prior_Tempering',
            'results': [
                {'lambda': 0.0, 'rmse': 4.86, 'mae': 3.89, 'r2': -0.15},
                {'lambda': 0.3, 'rmse': 5.12, 'mae': 4.08, 'r2': -0.21},
                {'lambda': 0.5, 'rmse': 5.21, 'mae': 4.15, 'r2': -0.24},
            ],
            'best': {'lambda': 0.0, 'rmse': 4.86, 'mae': 3.89, 'r2': -0.15}
        },
        'gam_ssm_lur_prior_tempering': {
            'experiment': 'GAM_SSM_LUR_Prior_Tempering',
            'results': [
                {'lambda': 0.0, 'rmse': 4.86, 'mae': 3.89, 'r2': -0.15},
                {'lambda': 0.3, 'rmse': 4.39, 'mae': 3.48, 'r2': -0.09},
                {'lambda': 0.5, 'rmse': 4.52, 'mae': 3.61, 'r2': -0.12},
            ],
            'best': {'lambda': 0.3, 'rmse': 4.39, 'mae': 3.48, 'r2': -0.09}
        },
        'fusiongp_obtl': {
            'experiment': 'FusionGP_OBTL',
            'results': [
                {'delta': 0.3, 'rmse': 4.71, 'mae': 3.76, 'r2': -0.08},
                {'delta': 0.5, 'rmse': 4.71, 'mae': 3.76, 'r2': -0.08},
            ],
            'best': {'delta': 0.3, 'rmse': 4.71, 'mae': 3.76, 'r2': -0.08}
        },
        'gam_ssm_lur_obtl': {
            'experiment': 'GAM_SSM_LUR_OBTL',
            'results': [
                {'delta': 0.3, 'rmse': 4.71, 'mae': 3.76, 'r2': -0.08},
                {'delta': 0.5, 'rmse': 4.71, 'mae': 3.76, 'r2': -0.08},
            ],
            'best': {'delta': 0.3, 'rmse': 4.71, 'mae': 3.76, 'r2': -0.08}
        }
    }


def test_export():
    """Test the table export functionality."""
    print("="*70)
    print("TESTING TABLE EXPORT FUNCTIONALITY")
    print("="*70)

    # Create test output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'test_table_export'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create test results
    print("\n1. Creating test results dictionary...")
    test_results = create_test_results()
    print("   ✓ Test results created")

    # Export tables
    print("\n2. Exporting tables...")
    try:
        export_results_tables(test_results, output_dir, 'test')
        print("   ✓ Tables exported successfully")
    except Exception as e:
        print(f"   ❌ Error exporting tables: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify files exist in tables/ subdirectory
    print("\n3. Verifying files exist in tables/ subdirectory...")
    tables_dir = output_dir / 'tables'
    if not tables_dir.exists():
        print(f"   ❌ tables/ subdirectory MISSING")
        return False

    print(f"   ✓ tables/ subdirectory exists")

    expected_files = [
        'prior_tempering_results.csv',
        'prior_tempering_results.tex',
        'obtl_results.csv',
        'obtl_results.tex',
        'summary_best_results.csv',
        'summary_best_results.tex'
    ]

    all_exist = True
    for filename in expected_files:
        filepath = tables_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"   ✓ {filename} ({size} bytes)")
        else:
            print(f"   ❌ {filename} MISSING")
            all_exist = False

    if not all_exist:
        return False

    # Verify CSV contents
    print("\n4. Verifying CSV contents...")
    try:
        df_pt = pd.read_csv(tables_dir / 'prior_tempering_results.csv')
        print(f"   ✓ Prior Tempering CSV: {len(df_pt)} rows, {len(df_pt.columns)} columns")
        print(f"     Columns: {', '.join(df_pt.columns)}")

        df_obtl = pd.read_csv(tables_dir / 'obtl_results.csv')
        print(f"   ✓ OBTL CSV: {len(df_obtl)} rows, {len(df_obtl.columns)} columns")

        df_summary = pd.read_csv(tables_dir / 'summary_best_results.csv')
        print(f"   ✓ Summary CSV: {len(df_summary)} rows, {len(df_summary.columns)} columns")
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return False

    # Verify LaTeX contents
    print("\n5. Verifying LaTeX contents...")
    try:
        tex_pt = (tables_dir / 'prior_tempering_results.tex').read_text()
        if '\\begin{table}' in tex_pt and '\\toprule' in tex_pt:
            print("   ✓ Prior Tempering LaTeX: Valid table structure")
        else:
            print("   ❌ Prior Tempering LaTeX: Missing table commands")
            return False

        tex_obtl = (tables_dir / 'obtl_results.tex').read_text()
        if 'OBTL' in tex_obtl and '\\bottomrule' in tex_obtl:
            print("   ✓ OBTL LaTeX: Valid table structure")
        else:
            print("   ❌ OBTL LaTeX: Missing OBTL or table commands")
            return False

        tex_summary = (tables_dir / 'summary_best_results.tex').read_text()
        if 'Best' in tex_summary:
            print("   ✓ Summary LaTeX: Valid table structure")
        else:
            print("   ❌ Summary LaTeX: Missing 'Best' in caption")
            return False
    except Exception as e:
        print(f"   ❌ Error reading LaTeX: {e}")
        return False

    # Show sample data
    print("\n6. Sample data from CSV:")
    print("\n   Prior Tempering (first 3 rows):")
    print(df_pt.head(3).to_string(index=False))

    print("\n   Summary:")
    print(df_summary.to_string(index=False))

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print(f"\nTest output saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Check the CSV files in Excel/Google Sheets")
    print("2. Verify LaTeX tables compile in your thesis")
    print("3. Run full experiment: python experiments/run_model_transfer_experiments.py")

    return True


if __name__ == '__main__':
    success = test_export()
    sys.exit(0 if success else 1)
