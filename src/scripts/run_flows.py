"""
Script to test and run Prefect flows locally.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.orchestration.flows import (
    data_ingestion_flow,
    drift_check_flow,
    retrain_flow,
    prediction_flow,
    performance_monitoring_flow,
    end_to_end_pipeline,
)


def main():
    """Run the end-to-end pipeline locally."""
    print("\n" + "=" * 70)
    print("MLOPS ENERGY FORECASTING - PREFECT PIPELINE")
    print("=" * 70)
    
    print("\nAvailable flows:")
    print("  1. Data Ingestion (fetch & process data)")
    print("  2. Drift Check (detect distribution shifts)")
    print("  3. Retrain Model (train new model)")
    print("  4. Generate Predictions (forecast demand)")
    print("  5. Performance Monitoring (evaluate recent performance)")
    print("  6. End-to-End Pipeline (complete workflow)")
    print("  0. Exit")
    
    while True:
        choice = input("\nSelect flow to run (0-6): ").strip()
        
        if choice == "0":
            print("Exiting...")
            break
        
        try:
            if choice == "1":
                print("\n" + "-" * 70)
                print("Running: Data Ingestion Flow")
                print("-" * 70)
                result = data_ingestion_flow(days_back=7, save_to_disk=True)
                print(f"\n✓ Ingested {len(result)} feature rows")
            
            elif choice == "2":
                print("\n" + "-" * 70)
                print("Running: Drift Check Flow")
                print("-" * 70)
                result = drift_check_flow(
                    days_back=7,
                    generate_report=True,
                    trigger_retrain=False  # Don't auto-retrain in interactive mode
                )
                if "error" not in result:
                    print(f"\n✓ Drift Share: {result['drift_share']:.2%}")
                    print(f"  Should Retrain: {result['should_retrain']}")
            
            elif choice == "3":
                days = input("Days of training data (default 60): ").strip() or "60"
                print("\n" + "-" * 70)
                print("Running: Retrain Flow")
                print("-" * 70)
                result = retrain_flow(
                    days_back=int(days),
                    use_cached_data=False,
                    auto_promote=True
                )
                print(f"\n✓ Run ID: {result['run_id']}")
                print(f"  MAE: {result['metrics']['mae']:.2f}")
                print(f"  Promoted: {result['promoted']}")
            
            elif choice == "4":
                hours = input("Forecast hours (default 24): ").strip() or "24"
                print("\n" + "-" * 70)
                print("Running: Prediction Flow")
                print("-" * 70)
                result = prediction_flow(
                    forecast_hours=int(hours),
                    save_predictions=True
                )
                if result is not None:
                    print(f"\n✓ Generated {len(result)} predictions")
            
            elif choice == "5":
                print("\n" + "-" * 70)
                print("Running: Performance Monitoring Flow")
                print("-" * 70)
                result = performance_monitoring_flow(days_back=7)
                if "error" not in result and result:
                    print(f"\n✓ MAE: {result.get('mae', 0):.2f}")
                    print(f"  RMSE: {result.get('rmse', 0):.2f}")
            
            elif choice == "6":
                print("\n" + "-" * 70)
                print("Running: End-to-End Pipeline")
                print("-" * 70)
                result = end_to_end_pipeline(
                    days_back=60,
                    check_drift=True,
                    retrain_on_drift=True
                )
                print("\n✓ Pipeline complete!")
                print(f"  Results: {result}")
            
            else:
                print("Invalid choice. Please select 0-6.")
                continue
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
