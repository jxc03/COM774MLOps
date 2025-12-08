import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

DATA_DIR = os.environ.get("DATA_DIR", "data")

def load_splits(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_classification.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test_classification.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_classification.csv"))
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test_classification.csv"))

    # If y has a single column, flatten it to a 1D array
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
        y_test  = y_test.iloc[:, 0]

    # ✅ NEW: keep only numeric / boolean feature columns
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns
    X_train = X_train[numeric_cols].copy()
    X_test  = X_test[numeric_cols].copy()

    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Simple but solid baseline
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    return model, acc, f1, classification_report(y_test, y_pred)

def main():
    mlflow.set_experiment("incident_priority_cw2")

    X_train, X_test, y_train, y_test = load_splits(DATA_DIR)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)

        model, acc, f1, clf_report = train_and_evaluate(
            X_train, X_test, y_train, y_test
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_text(clf_report, "classification_report.txt")

        mlflow.sklearn.log_model(model, name="model")

        print("Accuracy:", acc)
        print("F1 score:", f1)

if __name__ == "__main__":
    main()
