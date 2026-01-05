import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Import from code/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.append(str(CODE_DIR))

from data_util import load_train_test
from model_util import build_model


# Set a baseline
MIN_BASELINE_ACCURACY = 0.5

# Test to load real data, build model, train, evaluate & check minimum accuracy
class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = PROJECT_ROOT / "data_processed_split"
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_train_test(
            str(data_dir)
        )

        # Use a subset if full data is large
        cls.X_train_small = cls.X_train.iloc[:200].copy()
        cls.y_train_small = pd.Series(cls.y_train).iloc[:200].copy()
        cls.X_test_small = cls.X_test.iloc[:200].copy()
        cls.y_test_small = pd.Series(cls.y_test).iloc[:200].copy()

    def test_end_to_end_training_and_evaluation(self):
        model = build_model(
            X_sample=self.X_train_small,
            n_estimators=50,
            max_depth=None,
            random_state=42,
        )

        model.fit(self.X_train_small, self.y_train_small)
        y_pred = model.predict(self.X_test_small)

        acc = accuracy_score(self.y_test_small, y_pred)
        print(f"[Integration test] Accuracy on small test subset: {acc:.3f}")

        self.assertGreaterEqual(
            acc,
            MIN_BASELINE_ACCURACY,
            f"Accuracy {acc:.3f} is below baseline {MIN_BASELINE_ACCURACY}",
        )

        # Basic sanity: predictions shape
        self.assertEqual(len(y_pred), len(self.X_test_small))

    # Test that number of predictions matches number of input
    def test_predictions_shape_matches_input(self):
        model = build_model(
            X_sample=self.X_train_small,
            n_estimators=50,
            max_depth=None,
            random_state=42,
        )

        model.fit(self.X_train_small, self.y_train_small)
        y_pred = model.predict(self.X_test_small)

        self.assertEqual(
            len(y_pred), 
            len(self.X_test_small),
            "Number of predictions doesn't match number of test samples"
        )

    # Test that model produces consistent results with same random_state
    def test_model_performance_consistent_across_runs(self):
        model1 = build_model(X_sample=self.X_train_small, n_estimators=20, random_state=42)
        model2 = build_model(X_sample=self.X_train_small, n_estimators=20, random_state=42)
        
        model1.fit(self.X_train_small, self.y_train_small)
        model2.fit(self.X_train_small, self.y_train_small)
        
        pred1 = model1.predict(self.X_test_small)
        pred2 = model2.predict(self.X_test_small)
        
        np.testing.assert_array_equal(
            pred1, pred2,
            "Model predictions should be identical with same random_state"
        )

if __name__ == "__main__":
    unittest.main()
