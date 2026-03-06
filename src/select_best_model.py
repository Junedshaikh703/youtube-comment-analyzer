import mlflow
import json

import dagshub

dagshub.init(
    repo_owner="Junedshaikh703",
    repo_name="youtube-comment-analyzer",
    mlflow=True
)

EXPERIMENT_NAME = "Models Experimentation"

# ensure the experiment is created/active before querying
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found; have you run the pipeline?")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

best_score = -1
best_run = None

for _, run in runs.iterrows():

    summary = run["metrics.avg_final_score"]
    reply = run["metrics.avg_reply_final"]

    if summary is None or reply is None:
        continue

    combined_score = (summary + reply) / 2

    if combined_score > best_score:
        best_score = combined_score
        best_run = run

if best_run is None:
    raise RuntimeError(f"No completed runs with both summary and reply metrics were found in experiment '{EXPERIMENT_NAME}'")


best_model = best_run["params.model"]

best_config = {
    "provider": "groq",
    "model_name": best_model,
    "summary_temperature": 0,
    "reply_temperature": 0
}

with open("best_model_config.json", "w") as f:
    json.dump(best_config, f, indent=4)

print("Best model selected:", best_model)
print("Score:", best_score)