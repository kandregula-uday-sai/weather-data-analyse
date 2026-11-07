"""Train and test the student project's custom Random Forest model."""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from models.random_forest import RandomForest

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_data_shuffled.csv"
MODEL_PATH = BASE_DIR / "models" / "rain_prediction_model.pkl"
RESULTS_PATH = BASE_DIR / "model_results.json"
TARGET = "RainTomorrow"


def train_and_evaluate():
    print("=" * 58)
    print("WEATHER RAIN PREDICTION - MODEL TRAINING")
    print("=" * 58)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain {TARGET}")
    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values. Please clean them before training.")

    feature_names = [column for column in data.columns if column != TARGET]
    X = data[feature_names].to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Dataset: {len(data):,} rows | {len(feature_names)} input features")
    print(f"Training rows: {len(X_train):,} | Testing rows: {len(X_test):,}")
    print("Training 10 custom decision trees. This may take a few minutes...")
    model = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training finished in {time.time() - start:.1f} seconds.")

    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "test_records": int(len(y_test)),
        "train_records": int(len(y_train)),
        "feature_names": feature_names,
        "dataset_name": DATA_PATH.name,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTEST RESULTS")
    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print(f"F1 score : {metrics['f1']:.2%}")
    print("Confusion matrix [No Rain, Rain]:")
    print(matrix)
    print("\n", classification_report(y_test, predictions, target_names=["No Rain", "Rain"], zero_division=0))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
