import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Init MLflow + DagsHub
dagshub.init(repo_owner='eyacherif03', repo_name='MLOps-water-potability', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/eyacherif03/MLOps-water-potability.mlflow")

EXPERIMENT_NAME = "Water_Potability_Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)


def train():
    train_data = pd.read_csv("/app/dataset/processed/train.csv")
    test_data  = pd.read_csv("/app/dataset/processed/test.csv")

    X_train = train_data.drop(columns=["Potability"])
    y_train = train_data["Potability"]
    X_test  = test_data.drop(columns=["Potability"])
    y_test  = test_data["Potability"]

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "rf":      RandomForestClassifier(),
        "svc":     SVC(),
        "dt":      DecisionTreeClassifier(),
        "knn":     KNeighborsClassifier(),
        "xgb":     XGBClassifier()
    }

    with mlflow.start_run(run_name="training_pipeline"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                print(f"[INFO] Training {name}")

                # Train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
                mlflow.log_metric("precision", precision_score(y_test, y_pred))
                mlflow.log_metric("recall",    recall_score(y_test, y_pred))
                mlflow.log_metric("f1_score",  f1_score(y_test, y_pred))
                mlflow.log_param("model_name", name)

                pred_df = pd.DataFrame({
                    "y_true": y_test.values,
                    "y_pred": y_pred
                })

                pred_file = f"{name}_output.csv"
                pred_path = f"/tmp/{pred_file}"
                pred_df.to_csv(pred_path, index=False)

                mlflow.log_artifact(pred_path, artifact_path="outputs")
                model_dir = f"/tmp/{name}_model"

                # nettoyer si existe
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)

                mlflow.sklearn.save_model(model, model_dir)
                mlflow.log_artifacts(model_dir, artifact_path="model")

                print(f"Model + outputs logged for {name}")


if __name__ == "__main__":
    train()