import time
import mlflow

start = time.time()
client = mlflow.MlflowClient()
versions = client.search_model_versions("name='energy-demand-lgbm'", max_results=5)
latest = max(versions, key=lambda x: int(x.version))
run = client.get_run(latest.run_id)
elapsed = time.time() - start

print(f"Fetched model v{latest.version} in {elapsed:.2f}s")
print(f"MAE: {run.data.metrics.get('mae', 0):.0f}")
