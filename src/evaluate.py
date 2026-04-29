import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

dagshub.init(repo_owner='eyacherif03', repo_name='Water-potability', mlflow=True)

EXPERIMENT_NAME = "Water_Potability_Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

def evaluate():
    # ── Charger les données de test ──────────────────────────────────────────
    test_data = pd.read_csv("dataset/processed/test.csv")
    X_test    = test_data.drop(columns=["Potability"])
    y_test    = test_data["Potability"]

    # ── Récupérer tous les child runs depuis MLflow ──────────────────────────
    client     = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experience '{EXPERIMENT_NAME}' introuvable dans MLflow")

    # Récupérer uniquement les child runs (pas le run parent)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId != ''",  # child runs seulement
        
    )

    if not runs:
        raise ValueError("Aucun run trouvé — lancer train.py d'abord")

    os.makedirs("reports", exist_ok=True)

    best_run    = None
    best_f1     = -1
    all_metrics = []

    # ── Run parent pour regrouper toutes les évaluations ────────────────────
    with mlflow.start_run(run_name="Evaluation_Pipeline"):

        for run in runs:
            model_name = run.data.tags.get("mlflow.runName", "unknown")
            run_id     = run.info.run_id
            model_uri  = f"runs:/{run_id}/model"

            try:
                model = mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                print(f"Impossible de charger {model_name} : {e}")
                continue

            y_pred = model.predict(X_test)

            # Calculer les métriques
            acc       = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred)

            # Logguer dans un child run dédié à l'évaluation
            with mlflow.start_run(run_name=f"eval_{model_name}", nested=True):
                mlflow.log_metric("accuracy",  acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall",    recall)
                mlflow.log_metric("f1_score",  f1)

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(5, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"Confusion Matrix — {model_name}")
                plot_path = f"reports/cm_{model_name}.png"
                plt.savefig(plot_path)
                plt.close()
                mlflow.log_artifact(plot_path)

            print(f"  {model_name:<30} F1={f1:.4f}  Acc={acc:.4f}")

            all_metrics.append({
                "model":     model_name,
                "run_id":    run_id,
                "accuracy":  acc,
                "precision": precision,
                "recall":    recall,
                "f1_score":  f1
            })

            # Garder le meilleur modèle
            if f1 > best_f1:
                best_f1  = f1
                best_run = run

        # ── Enregistrer le meilleur modèle dans le Registry ─────────────────
        if best_run is None:
            raise ValueError("Aucun modèle valide trouvé")

        best_model_name = best_run.data.tags.get("mlflow.runName", "unknown")
        best_model_uri  = f"runs:/{best_run.info.run_id}/model"

        mlflow.register_model(best_model_uri, "WaterPotabilityModel")
        print(f"\n Meilleur modèle enregistré : {best_model_name} (F1={best_f1:.4f})")

        # ── Sauvegarder best_model.json pour app.py ──────────────────────────
        result = {
            "best_model_name": best_model_name,
            "best_run_id":     best_run.info.run_id,
            "model_uri":       best_model_uri,
            "metrics": {
                "accuracy":  next(m["accuracy"]  for m in all_metrics if m["model"] == best_model_name),
                "precision": next(m["precision"] for m in all_metrics if m["model"] == best_model_name),
                "recall":    next(m["recall"]    for m in all_metrics if m["model"] == best_model_name),
                "f1_score":  best_f1
            },
            "all_models": all_metrics
        }

        with open("reports/best_model.json", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    evaluate()