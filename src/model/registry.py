"""
Helpers for MLflow Model Registry — champion/challenger pattern.
Falls back to MLflow run tags if Model Registry is unavailable (e.g. new DagsHub repo).
"""
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Tag used as fallback when Model Registry is unavailable
CHAMPION_TAG = "champion"
CHAMPION_EXPERIMENT_TAG = "is_champion"


def _registry_available() -> bool:
    """Check if the MLflow Model Registry is available on this tracking server."""
    try:
        client.search_registered_models(max_results=1)
        return True
    except MlflowException:
        return False


def _get_champion_run_id_from_tags() -> str | None:
    """Fallback: find the champion run by searching experiments for the is_champion tag."""
    try:
        runs = mlflow.search_runs(
            filter_string=f"tags.{CHAMPION_EXPERIMENT_TAG} = 'true'",
            order_by=["metrics.mae ASC"],
            max_results=1,
        )
        if not runs.empty:
            return runs.iloc[0]["run_id"]
    except Exception:
        pass
    return None


def _get_best_run_id_by_mae() -> str | None:
    """Fallback: find the run with the lowest MAE across all experiments."""
    try:
        runs = mlflow.search_runs(
            order_by=["metrics.mae ASC"],
            max_results=1,
        )
        if not runs.empty:
            return runs.iloc[0]["run_id"]
    except Exception:
        pass
    return None


def get_champion_model():
    """
    Load the current champion model.
    Tries Model Registry first, falls back to run tags, then best MAE run.
    """
    # Try Model Registry
    if _registry_available():
        try:
            model_uri = f"models:/{MODEL_NAME}@champion"
            return mlflow.lightgbm.load_model(model_uri)
        except MlflowException:
            try:
                model_uri = f"models:/{MODEL_NAME}/latest"
                return mlflow.lightgbm.load_model(model_uri)
            except MlflowException:
                pass

    # Fallback: use run tags
    print("Model Registry unavailable or empty — falling back to run tags.")
    run_id = _get_champion_run_id_from_tags() or _get_best_run_id_by_mae()
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        print(f"Loading model from run: {run_id}")
        return mlflow.lightgbm.load_model(model_uri)

    raise ValueError(
        f"No model found. Please train a model first."
    )


def get_latest_model_version():
    """Get the latest version number of the registered model."""
    if not _registry_available():
        return None
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))
        return latest.version
    except MlflowException:
        return None


def promote_to_champion(run_id: str) -> bool:
    """
    Compare the new model against the current champion.
    Promotes via Model Registry if available, otherwise uses MLflow run tags.
    Returns True if promoted.
    """
    new_run = client.get_run(run_id)
    new_mae = float(new_run.data.metrics.get("mae", float("inf")))

    if _registry_available():
        # --- Model Registry path ---
        try:
            champion_versions = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champion_run_id = champion_versions.run_id
            champion_run = client.get_run(champion_run_id)
            champion_mae = float(champion_run.data.metrics.get("mae", float("inf")))
        except MlflowException:
            champion_mae = float("inf")
            print("No existing champion found in registry. Promoting automatically.")

        if new_mae < champion_mae:
            try:
                versions = client.search_model_versions(f"run_id='{run_id}'")
                if versions:
                    version = versions[0].version
                    client.set_registered_model_alias(MODEL_NAME, "champion", version)
                    print(f"Promoted version {version} to champion (MAE: {new_mae:.1f} < {champion_mae:.1f})")
                    return True
            except MlflowException as e:
                print(f"Registry promotion failed: {e} — falling back to tags.")
        else:
            print(f"New model NOT promoted (MAE: {new_mae:.1f} >= {champion_mae:.1f})")
            return False

    # --- Tag-based fallback path ---
    print("Using tag-based champion tracking (Model Registry unavailable).")
    champion_run_id = _get_champion_run_id_from_tags()
    champion_mae = float("inf")
    if champion_run_id:
        try:
            champion_run = client.get_run(champion_run_id)
            champion_mae = float(champion_run.data.metrics.get("mae", float("inf")))
        except Exception:
            pass

    if new_mae < champion_mae:
        # Clear old champion tag
        if champion_run_id:
            try:
                client.delete_tag(champion_run_id, CHAMPION_EXPERIMENT_TAG)
            except Exception:
                pass
        # Set new champion tag
        client.set_tag(run_id, CHAMPION_EXPERIMENT_TAG, "true")
        client.set_tag(run_id, "model_alias", "champion")
        print(f"Promoted run {run_id} to champion via tags (MAE: {new_mae:.1f} < {champion_mae:.1f})")
        return True
    else:
        print(f"New model NOT promoted (MAE: {new_mae:.1f} >= {champion_mae:.1f})")
        return False


def get_model_by_version(version: str):
    """Load a specific model version."""
    model_uri = f"models:/{MODEL_NAME}/{version}"
    return mlflow.lightgbm.load_model(model_uri)


def list_all_model_versions():
    """List all registered model versions with their metrics."""
    if not _registry_available():
        # Fallback: list runs ordered by MAE
        try:
            runs = mlflow.search_runs(order_by=["metrics.mae ASC"], max_results=20)
            version_info = []
            for i, row in runs.iterrows():
                version_info.append({
                    "version": str(i + 1),
                    "run_id": row["run_id"],
                    "stage": "Champion" if row.get(f"tags.{CHAMPION_EXPERIMENT_TAG}") == "true" else "None",
                    "mae": row.get("metrics.mae"),
                    "rmse": row.get("metrics.rmse"),
                    "r2": row.get("metrics.r2"),
                    "creation_timestamp": None,
                })
            return version_info
        except Exception:
            return []

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
    """Archive old model versions, keeping only the latest N."""
    if not _registry_available():
        print("Model Registry unavailable — skipping archive.")
        return

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
