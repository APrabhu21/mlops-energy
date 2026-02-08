"""
Helpers for MLflow Model Registry — champion/challenger pattern.
"""
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def get_champion_model():
    """
    Load the current Production (champion) model.
    Falls back to latest version if no champion is set.
    """
    try:
        # Try to get the model with 'champion' alias
        model_uri = f"models:/{MODEL_NAME}@champion"
        return mlflow.lightgbm.load_model(model_uri)
    except MlflowException:
        # Fallback: get latest version
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            return mlflow.lightgbm.load_model(model_uri)
        except MlflowException:
            raise ValueError(
                f"No model found with name '{MODEL_NAME}'. "
                "Please train a model first."
            )


def get_latest_model_version():
    """Get the latest version number of the registered model."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            return None
        # Sort by version number and get the highest
        latest = max(versions, key=lambda v: int(v.version))
        return latest.version
    except MlflowException:
        return None


def promote_to_champion(run_id: str) -> bool:
    """
    Compare the new model (from run_id) against the current champion.
    Promote if the new model has lower MAE.

    Returns True if promoted.
    """
    # Get the new model's MAE
    new_run = client.get_run(run_id)
    new_mae = float(new_run.data.metrics.get("mae", float("inf")))

    # Get the current champion's MAE
    try:
        champion_versions = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run_id = champion_versions.run_id
        champion_run = client.get_run(champion_run_id)
        champion_mae = float(champion_run.data.metrics.get("mae", float("inf")))
    except MlflowException:
        # No champion exists yet — promote automatically
        champion_mae = float("inf")
        print("No existing champion found. Promoting new model automatically.")

    if new_mae < champion_mae:
        # Get the latest version number for this run
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            version = versions[0].version
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            print(f"✓ Promoted version {version} to champion (MAE: {new_mae:.1f} < {champion_mae:.1f})")
            return True
    else:
        print(f"✗ New model NOT promoted (MAE: {new_mae:.1f} >= {champion_mae:.1f})")
        return False


def get_model_by_version(version: str):
    """Load a specific model version."""
    model_uri = f"models:/{MODEL_NAME}/{version}"
    return mlflow.lightgbm.load_model(model_uri)


def list_all_model_versions():
    """List all registered model versions with their metrics."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        version_info = []
        for v in versions:
            run = client.get_run(v.run_id)
            info = {
                "version": v.version,
                "run_id": v.run_id,
                "stage": v.current_stage,
                "mae": run.data.metrics.get("mae"),
                "rmse": run.data.metrics.get("rmse"),
                "r2": run.data.metrics.get("r2"),
                "creation_timestamp": v.creation_timestamp,
            }
            version_info.append(info)
        
        return sorted(version_info, key=lambda x: int(x["version"]), reverse=True)
    except MlflowException:
        return []


def archive_old_versions(keep_latest: int = 5):
    """
    Archive old model versions, keeping only the latest N.
    
    Args:
        keep_latest: Number of recent versions to keep
    """
    versions = list_all_model_versions()
    
    if len(versions) <= keep_latest:
        print(f"Only {len(versions)} versions exist. Nothing to archive.")
        return
    
    to_archive = versions[keep_latest:]
    
    for v in to_archive:
        version_num = v["version"]
        try:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version_num,
                stage="Archived"
            )
            print(f"Archived version {version_num}")
        except MlflowException as e:
            print(f"Could not archive version {version_num}: {e}")
