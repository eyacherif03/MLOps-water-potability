import mlflow.sklearn
import json, os

def load_best_model():
    json_path = "reports/best_model.json"
    with open(json_path) as f:
        info = json.load(f)

    # Charger depuis MLflow Registry directement
    model = mlflow.sklearn.load_model(
        f"models:/WaterPotabilityModel/latest"
    )
    return model, info