import mlflow
import mlflow.sklearn
import dagshub
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
dagshub.init(repo_owner='eyacherif03', repo_name='MLOps-water-potability', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/eyacherif03/MLOps-water-potability.mlflow")
EXPERIMENT_NAME = "Water_Potability_Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)


def evaluate():
    client     = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' introuvable")

    # Récupérer les child runs avec leurs métriques déjà loggées
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId != ''",
        order_by=["metrics.f1_score DESC"]  # tri direct par F1
    )

    if not runs:
        raise ValueError("Aucun run trouvé — lancer train.py d'abord")

    os.makedirs("/app/reports", exist_ok=True)

    # Meilleur modèle = premier run (déjà trié par F1)
    best_run   = runs[0]
    best_f1    = best_run.data.metrics.get("f1_score", 0)
    best_name  = best_run.data.tags.get("mlflow.runName", "unknown")

    # Générer confusion matrices pour tous les modèles
    test_data = pd.read_csv("/app/dataset/processed/test.csv")
    X_test    = test_data.drop(columns=["Potability"])
    y_test    = test_data["Potability"]

    all_metrics = []

    for run in runs:
        model_name = run.data.tags.get("mlflow.runName", "unknown")
        model      = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
        y_pred     = model.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix — {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"/app/reports/cm_{model_name}.png")
        plt.close()

        all_metrics.append({
            "model":     model_name,
            "run_id":    run.info.run_id,
            "f1_score":  run.data.metrics.get("f1_score"),
            "accuracy":  run.data.metrics.get("accuracy"),
            "precision": run.data.metrics.get("precision"),
            "recall":    run.data.metrics.get("recall")
        })

    # Enregistrer le meilleur dans le Registry
    mlflow.register_model(f"runs:/{best_run.info.run_id}/model", "WaterPotabilityModel")
    print(f"Meilleur modèle : {best_name} (F1={best_f1:.4f})")

    # Sauvegarder best_model.json
    result = {
        "best_model_name": best_name,
        "best_run_id":     best_run.info.run_id,
        "model_uri":       f"runs:/{best_run.info.run_id}/model",
        "metrics":         next(m for m in all_metrics if m["model"] == best_name),
        "all_models":      all_metrics
    }

    with open("/app/reports/best_model.json", "w") as f:
        json.dump(result, f, indent=4)

    print("Evaluation done")

if __name__ == "__main__":
    evaluate()