from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def build_model(
    X_sample,
    n_estimators=100,
    max_depth=None,
    random_state=42,
):
    
    # To help fix error related to categorical features (3 - moderate)
    # Identify numeric vs categorical columns
    numeric_features = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_sample.select_dtypes(exclude=[np.number]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Preprocessing for numeric and categorical data
    numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False # scikit-learn >= 1.2; if older, use sparse=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    # Full pipeline: preprocessing + classifier
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    return model
