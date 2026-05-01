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
import os
token = os.getenv("DAGSHUB_TOKEN")


dagshub.init(repo_owner='eyacherif03', repo_name='Water-potability', mlflow=True, token=token)
mlflow.set_experiment("Water_Potability_Experiment")

def train():
    train_data = pd.read_csv("dataset/processed/train.csv")

    X_train = train_data.drop(columns=["Potability"])
    y_train = train_data["Potability"]

    models = {
        "log_reg": LogisticRegression(),
        "rf": RandomForestClassifier(),
        "svc": SVC(),
        "dt": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier()
    }

    with mlflow.start_run(run_name="training_pipeline"):
        for name, model in models.items():

            with mlflow.start_run(run_name=name, nested=True):
                model.fit(X_train, y_train)

                mlflow.sklearn.log_model(model, artifact_path="model")

                mlflow.log_param("model_name", name)

    print("Training done")

if __name__ == "__main__":
    train()