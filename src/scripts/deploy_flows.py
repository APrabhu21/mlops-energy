"""
Deploy Prefect flows with schedules.
This script creates deployments for automated execution.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from prefect import serve
from prefect.deployments import run_deployment
from prefect.schedules import IntervalSchedule, CronSchedule
from datetime import timedelta

from src.orchestration.flows import (
    data_ingestion_flow,
    drift_check_flow,
    performance_monitoring_flow,
)


def create_deployments():
    """
    Create Prefect deployments with schedules.
    
    Schedules:
    - Data Ingestion: Every 6 hours
    - Drift Check: Daily at 2 AM
    - Performance Monitoring: Daily at 3 AM
    """
    print("Creating Prefect deployments with schedules...\n")
    
    # Data Ingestion: Every 6 hours
    data_ingestion_deployment = data_ingestion_flow.to_deployment(
        name="data-ingestion-scheduled",
        schedule=IntervalSchedule(interval=timedelta(hours=6)),
        parameters={
            "days_back": 7,
            "save_to_disk": True
        },
        description="Fetch and process data from EIA and Weather APIs every 6 hours",
        tags=["production", "data-ingestion"]
    )
    
    # Drift Check: Daily at 2 AM
    drift_check_deployment = drift_check_flow.to_deployment(
        name="drift-check-daily",
        schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
        parameters={
            "days_back": 7,
            "generate_report": True,
            "trigger_retrain": True  # Auto-retrain if drift detected
        },
        description="Check for data drift daily and trigger retraining if needed",
        tags=["production", "drift-detection", "monitoring"]
    )
    
    # Performance Monitoring: Daily at 3 AM
    performance_deployment = performance_monitoring_flow.to_deployment(
        name="performance-monitoring-daily",
        schedule=CronSchedule(cron="0 3 * * *"),  # 3 AM daily
        parameters={
            "days_back": 7
        },
        description="Monitor model performance on recent predictions",
        tags=["production", "monitoring", "performance"]
    )
    
    print("✓ Created 3 deployments:")
    print("  1. data-ingestion-scheduled (every 6 hours)")
    print("  2. drift-check-daily (2 AM daily)")
    print("  3. performance-monitoring-daily (3 AM daily)")
    print("\nDeployments will run automatically according to their schedules.")
    print("\nTo start the worker, run:")
    print("  python src/scripts/deploy_flows.py --serve")
    
    return [
        data_ingestion_deployment,
        drift_check_deployment,
        performance_deployment,
    ]


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Prefect flows")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start serving the deployments (blocks until interrupted)"
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Run a specific deployment by name"
    )
    
    args = parser.parse_args()
    
    deployments = create_deployments()
    
    if args.serve:
        print("\n" + "=" * 70)
        print("Starting Prefect worker to serve deployments...")
        print("Press Ctrl+C to stop")
        print("=" * 70 + "\n")
        
        # Serve all deployments
        serve(*deployments)
    
    elif args.run:
        print(f"\nRunning deployment: {args.run}")
        run_deployment(name=args.run)
    
    else:
        print("\nDeployments created but not started.")
        print("Use --serve to start the worker, or --run <name> to run a deployment once.")


if __name__ == "__main__":
    main()
