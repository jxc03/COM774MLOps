import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature

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

    # Log training parameters (hyperparameters)
    with mlflow.start_run():
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
    report_path.write_text(report_str)

    # Feature importances
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

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(range(n_top), importances[indices][::-1])
        ax2.set_yticks(range(n_top))
        ax2.set_yticklabels(feature_names[indices][::-1])
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("Top 20 Feature Importances (RandomForest)")
        fig2.tight_layout()

        fi_path = output_dir / "feature_importances_top20.png"
        fig2.savefig(fi_path, dpi=150)
        plt.close(fig2)
    
    mlflow.log_artifact(str(fi_path))
    X_sig = X_train.head(50) if hasattr(X_train, "head") else X_train[:50]
    y_sig = model.predict(X_sig)
    signature = infer_signature(X_sig, y_sig)

    # Create input example for model signature
    input_example = X_train.head(1) if hasattr(X_train, "head") else X_train[:1]
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    print("Training Complete")

if __name__ == "__main__":
    main()
