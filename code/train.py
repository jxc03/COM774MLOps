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
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

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
    print(f"\nModel Evaluation")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
          
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
    print(f"\nClassification Report\n{report_str}")
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)
    mlflow.log_artifact(str(report_path))

    # Feature importances
    import numpy as np
    import joblib
    # Access the classifier step in the pipeline
    clf = model.named_steps['clf']
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_

        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback if get_feature_names_out not available
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        n_top = min(20, len(importances))
        indices = np.argsort(importances)[::-1][:n_top]
        top_importances = importances[indices]
        top_features = [feature_names[i] for i in indices]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(range(n_top), top_importances[::-1], color='steelblue')
        ax2.set_yticks(range(n_top))
        ax2.set_yticklabels(top_features[::-1])
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("Top 20 Feature Importances (RandomForest)")
        fig2.tight_layout()

        fi_path = output_dir / "feature_importances_top20.png"
        fig2.savefig(fi_path, dpi=150)
        plt.close(fig2)
        mlflow.log_artifact(str(fi_path))

    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path) # pip install joblib
    print(f"\nModel saved to: {model_path}")
    
    # Log the model directory as an artifact
    mlflow.log_artifacts(str(model_dir), artifact_path="model")

    # Try-except to handle Azure ML compatibility issues
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )
        print("MLflow sklearn model logged successfully")
    except Exception as e:
        print(f"Note: mlflow.sklearn.log_model skipped due to Azure ML compatibility: {e}")
        print("Model was saved as artifact instead (outputs/model/model.pkl)")

    print("\n=== Training Complete ===")

    # Log the trained model as an artefact
    # Later register & deploy
    #mlflow.sklearn.log_model(
        #sk_model=model,
        #artifact_path="model"
    #)

if __name__ == "__main__":
    main()
