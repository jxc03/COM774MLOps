import argparse

import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_util import load_train_test
from model_util import build_model

# For confusion matrix, classification report & bar chart
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    classification_report,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    # Set up the MLflow experiment
    #mlflow.set_experiment("incident_priority_experiment")

    # Everything inside this context gets logged as a single MLflow run
    #with mlflow.start_run():

    # Log training parameters (hyperparameters + any config)
    mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
            "data_dir": args.data_dir
        })

    # Load data
    X_train, X_test, y_train, y_test = load_train_test(args.data_dir)

    # Build and train the model
    model = build_model(
            X_sample=X_train, # pass X_train so build_model can inspect dtypes and build the ColumnTransformer
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)

    # Further artifacts: confusion matrix, classification report & feature importances
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Confusion matrix - incident priority")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    # Classification report
    report_str = classification_report(y_test, y_pred)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)
    mlflow.log_artifact(str(report_path))

    # Feature importances
    import numpy as np
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X_train.columns
        n_top = min(20, len(importances))
        indices = np.argsort(importances)[::-1][:n_top]
        top_importances = importances[indices]
        top_features = feature_names[indices]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(range(n_top), top_importances)
        ax2.set_xticks(range(n_top))
        ax2.set_xticklabels(top_features, rotation=90)
        ax2.set_ylabel("Feature importance")
        ax2.set_title("Top feature importances (RandomForest)")
        fig2.tight_layout()

        fi_path = output_dir / "feature_importances_top20.png"
        fig2.savefig(fi_path)
        plt.close(fig2)
        mlflow.log_artifact(str(fi_path))

    # Log the trained model as an artefact
    # Later register & deploy
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )


if __name__ == "__main__":
    main()
