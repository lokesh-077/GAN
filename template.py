import os

folders = [
    "artifacts/data_ingestion",
    "artifacts/data_transformation",
    "artifacts/data_validation",
    "artifacts/model_evaluation",
    "artifacts/model_trainer",
    "logs",
    "mlruns",
    "research",
    "src/datascience/components",
    "src/datascience/config",
    "src/datascience/constants",
    "src/datascience/entity",
    "src/datascience/pipeline",
    "src/datascience/utils",
    "templates"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, "__init__.py"), "a").close()

print("Project structure created ✅")
