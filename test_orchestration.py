"""
Quick test of individual Prefect flows.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.orchestration.flows import data_ingestion_flow, drift_check_flow


def test_data_ingestion():
    """Test data ingestion flow."""
    print("\n" + "=" * 70)
    print("TEST 1: Data Ingestion Flow")
    print("=" * 70)
    
    result = data_ingestion_flow(days_back=7, save_to_disk=True)
    
    print(f"\n✓ Success! Ingested {len(result)} feature rows")
    print(f"  Columns: {len(result.columns)}")
    return result


def test_drift_check():
    """Test drift check flow (without auto-retrain)."""
    print("\n" + "=" * 70)
    print("TEST 2: Drift Check Flow")
    print("=" * 70)
    
    result = drift_check_flow(
        days_back=7,
        generate_report=True,
        trigger_retrain=False  # Don't auto-retrain in test
    )
    
    if "error" in result:
        print(f"\n⚠️  Warning: {result['error']}")
    else:
        print(f"\n✓ Success!")
        print(f"  Drift Share: {result['drift_share']:.2%}")
        print(f"  Should Retrain: {result['should_retrain']}")
        print(f"  Drifted Features: {result['details']['n_drifted_features']}/{result['details']['n_total_features']}")
    
    return result


def main():
    """Run tests sequentially."""
    print("\n" + "=" * 70)
    print("MLOPS ORCHESTRATION TESTS")
    print("=" * 70)
    
    try:
        # Test 1: Data Ingestion
        features = test_data_ingestion()
        
        # Test 2: Drift Check
        drift_result = test_drift_check()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
