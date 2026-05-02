import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dagshub.init(repo_owner='eyacherif03', repo_name='MLOps-water-potability', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/eyacherif03/MLOps-water-potability.mlflow")
mlflow.set_experiment("Water_Potability_Experiment")


def train():
    train_data = pd.read_csv("/app/dataset/processed/train.csv")
    test_data  = pd.read_csv("/app/dataset/processed/test.csv")

    X_train = train_data.drop(columns=["Potability"])
    y_train = train_data["Potability"]
    X_test  = test_data.drop(columns=["Potability"])
    y_test  = test_data["Potability"]

    models = {
        "log_reg": LogisticRegression(),
        "rf":      RandomForestClassifier(),
        "svc":     SVC(),
        "dt":      DecisionTreeClassifier(),
        "knn":     KNeighborsClassifier(),
        "xgb":     XGBClassifier()
    }

    with mlflow.start_run(run_name="training_pipeline") as parent:
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # Entraînement
                model.fit(X_train, y_train)

                # Évaluation
                y_pred = model.predict(X_test)
                mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
                mlflow.log_metric("precision", precision_score(y_test, y_pred))
                mlflow.log_metric("recall",    recall_score(y_test, y_pred))
                mlflow.log_metric("f1_score",  f1_score(y_test, y_pred))

                # Sauvegarde modèle
                mlflow.log_param("model_name", name)
                mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train()