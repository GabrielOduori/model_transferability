"""
Test Refactored Modules
========================

Test the new configuration and data modules.
"""

import sys
from pathlib import Path

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))

from config import ExperimentConfig, DEFAULT_CONFIG, QUICK_TEST_CONFIG
from data import SyntheticDataGenerator, ModelLoader


def test_configuration():
    """Test configuration module."""
    print("\n" + "="*70)
    print("TEST 1: Configuration Module")
    print("="*70)

    # Test default config
    config = ExperimentConfig()
    print(f"\n✓ Created default config:")
    print(config)

    # Test predefined configs
    print(f"\n✓ Quick test config: {QUICK_TEST_CONFIG.n_target} target samples")

    # Test to_dict
    config_dict = config.to_dict()
    print(f"\n✓ Serialized to dict with {len(config_dict)} keys")

    # Test from_dict
    config2 = ExperimentConfig.from_dict(config_dict)
    assert config2.n_target == config.n_target
    print(f"✓ Deserialized successfully")

    print("\n✅ Configuration module: PASSED")
    return True


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\n" + "="*70)
    print("TEST 2: Synthetic Data Generator")
    print("="*70)

    # Create generator
    gen = SyntheticDataGenerator(seed=42)
    print(f"\n✓ Created data generator (seed=42)")

    # Generate data
    data = gen.generate(n_target=10, n_test=20, force_regenerate=True)
    print(f"✓ Generated synthetic data")

    # Check structure
    assert 'target' in data, "Missing 'target' key"
    assert 'test' in data, "Missing 'test' key"
    assert 'metadata' in data, "Missing 'metadata' key"
    print(f"✓ Data structure correct")

    # Check shapes
    assert data['target']['X'].shape == (10, 3), f"Wrong target X shape: {data['target']['X'].shape}"
    assert data['target']['y'].shape == (10,), f"Wrong target y shape: {data['target']['y'].shape}"
    assert data['test']['X'].shape == (20, 3), f"Wrong test X shape: {data['test']['X'].shape}"
    assert data['test']['y'].shape == (20,), f"Wrong test y shape: {data['test']['y'].shape}"
    print(f"✓ Data shapes correct:")
    print(f"  Target: X={data['target']['X'].shape}, y={data['target']['y'].shape}")
    print(f"  Test:   X={data['test']['X'].shape}, y={data['test']['y'].shape}")

    # Test reproducibility (load saved data)
    data2 = gen.generate(n_target=10, n_test=20, force_regenerate=False)
    print(f"✓ Loaded saved data (reproducibility)")
    assert data2['metadata']['loaded_from_file'] == True
    print(f"  File: {data2['metadata']['saved_file']}")

    print("\n✅ Synthetic data generator: PASSED")
    return True


def test_model_loader():
    """Test model loader."""
    print("\n" + "="*70)
    print("TEST 3: Model Loader")
    print("="*70)

    config = ExperimentConfig()

    # Verify model files
    loader = ModelLoader()
    status = loader.verify_model_files(config)

    all_exist = all(status.values())

    if all_exist:
        print("\n✓ All model files found")

        # Try loading FusionGP
        try:
            print("\n🔄 Testing FusionGP loading...")
            model, likelihood = loader.load_fusiongp(config.fusiongp_path)
            print(f"✓ FusionGP loaded successfully")
            print(f"  Model: {type(model).__name__}")
            print(f"  Likelihood: {type(likelihood).__name__}")
        except Exception as e:
            print(f"⚠️  FusionGP loading failed: {e}")
            return False

        # Try loading GAM-SSM-LUR
        try:
            print("\n🔄 Testing GAM-SSM-LUR loading...")
            gam_model, gam_data = loader.load_gam_ssm_lur(
                config.gam_path,
                config.ssm_path,
                config.gam_data_path
            )
            print(f"✓ GAM-SSM-LUR loaded successfully")
            print(f"  Model type: {gam_model['type']}")
            print(f"  Training samples: {gam_data['X_train'].shape[0]}")
        except Exception as e:
            print(f"⚠️  GAM-SSM-LUR loading failed: {e}")
            return False

        print("\n✅ Model loader: PASSED")
        return True
    else:
        print("\n⚠️  Some model files missing:")
        for name, exists in status.items():
            if not exists:
                print(f"  ✗ {name}")
        print("\n⚠️  Model loader: SKIPPED (missing files)")
        return None  # Skipped, not failed


def test_integration():
    """Test integration of all modules."""
    print("\n" + "="*70)
    print("TEST 4: Integration Test")
    print("="*70)

    # Use quick test config
    config = QUICK_TEST_CONFIG
    print(f"\n✓ Using QUICK_TEST_CONFIG:")
    print(f"  n_target={config.n_target}, n_test={config.n_test}")
    print(f"  lambda={config.lambda_values}")

    # Generate data
    gen = SyntheticDataGenerator(seed=config.seed)
    data = gen.generate(
        n_target=config.n_target,
        n_test=config.n_test,
        force_regenerate=True
    )
    print(f"\n✓ Generated test data")

    # Load models (if available)
    loader = ModelLoader()
    status = loader.verify_model_files(config)

    if all(status.values()):
        try:
            fusiongp_model, fusiongp_likelihood = loader.load_fusiongp(config.fusiongp_path)
            print(f"\n✓ Loaded FusionGP model")

            gam_model, gam_data = loader.load_gam_ssm_lur(
                config.gam_path,
                config.ssm_path,
                config.gam_data_path
            )
            print(f"✓ Loaded GAM-SSM-LUR model")

            print("\n✅ Integration test: PASSED")
            print("   All modules work together correctly!")
            return True

        except Exception as e:
            print(f"\n❌ Integration test: FAILED")
            print(f"   Error: {e}")
            return False
    else:
        print("\n⚠️  Integration test: SKIPPED (missing model files)")
        return None


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING REFACTORED MODULES")
    print("="*70)

    results = []

    # Run tests
    results.append(("Configuration", test_configuration()))
    results.append(("Synthetic Data", test_synthetic_data()))
    results.append(("Model Loader", test_model_loader()))
    results.append(("Integration", test_integration()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        if result is True:
            print(f"  ✅ {name}: PASSED")
        elif result is False:
            print(f"  ❌ {name}: FAILED")
        else:
            print(f"  ⚠️  {name}: SKIPPED")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ SOME TESTS FAILED")
        return 1
    elif passed > 0:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n⚠️  ALL TESTS SKIPPED")
        return 2


if __name__ == '__main__':
    exit(main())
