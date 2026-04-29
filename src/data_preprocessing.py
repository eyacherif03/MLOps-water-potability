import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib

def preprocess():
    DATA_PATH = os.getenv("DATA_PATH", "dataset/raw/water_potability.csv")

    data = pd.read_csv(DATA_PATH)

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=0
    )

    # median imputation
    median_values = train_data.median(numeric_only=True)

    train_data = train_data.fillna(median_values)
    test_data = test_data.fillna(median_values)

    os.makedirs("dataset/processed", exist_ok=True)

    train_data.to_csv("dataset/processed/train.csv", index=False)
    test_data.to_csv("dataset/processed/test.csv", index=False)

    # sauvegarde propre (pour inference)
    joblib.dump(median_values, "dataset/processed/median_values.pkl")

    print("Preprocessing done")

if __name__ == "__main__":
    preprocess()