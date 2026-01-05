import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

# Import from code/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.append(str(CODE_DIR))

from data_util import load_train_test
from model_util import build_model

# Test for the model builder: returns a Pipeline, can fit/predict on small data
class TestModelBuilding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = PROJECT_ROOT / "data_processed_split"
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_train_test(
            str(data_dir)
        )

    # Test that build_model returns a sklearn Pipeline
    def test_build_model_returns_pipeline(self):
        model = build_model(
            X_sample=self.X_train,
            n_estimators=10, # Small for test speed
            max_depth=5,
            random_state=42,
        )
        self.assertIsInstance(
            model, Pipeline, "build_model should return a sklearn Pipeline"
        )

    # Test that the model can fit and predict
    def test_model_can_fit_and_predict_on_small_subset(self):
        model = build_model(
            X_sample=self.X_train,
            n_estimators=10,
            max_depth=5,
            random_state=42,
        )

        # Use a small subset for speed
        X_small = self.X_train.iloc[:50]
        y_small = pd.Series(self.y_train).iloc[:50]

        model.fit(X_small, y_small)
        preds = model.predict(X_small)

        self.assertEqual(
            len(preds),
            len(X_small),
            "Number of predictions should match number of samples",
        )

        # Check predictions are finite numbers or valid classes
        self.assertFalse(
            np.any(pd.isna(preds)), "Predictions contain NaNs"
        )

    # Test that predictions are valid class labels
    def test_model_predictions_are_valid_classes(self):
        model = build_model(
            X_sample=self.X_train,
            n_estimators=10,
            max_depth=5,
            random_state=42,
        )

        X_small = self.X_train.iloc[:50]
        y_small = pd.Series(self.y_train).iloc[:50]
        
        model.fit(X_small, y_small)
        preds = model.predict(X_small)
        
        # All predictions should be in the set of known classes
        known_classes = set(y_small.unique())
        pred_classes = set(preds)
        self.assertTrue(
            pred_classes.issubset(known_classes),
            f"Predictions {pred_classes} contain unknown classes (expected {known_classes})"
        )

if __name__ == "__main__":
    unittest.main()
