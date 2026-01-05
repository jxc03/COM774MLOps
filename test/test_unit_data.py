import unittest
from pathlib import Path
import sys

import pandas as pd

# Import from code/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.append(str(CODE_DIR))

from data_util import load_train_test  

# Test for data loading, allignment and dtypes
class TestDataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path to the folder that contains the CSVs
        cls.data_dir = PROJECT_ROOT / "data_processed_split"
        if not cls.data_dir.exists():
            raise RuntimeError(f"Data directory not found: {cls.data_dir}")

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_train_test(
            str(cls.data_dir)
        )

    # Test that all datasets have data
    def test_train_test_not_empty(self):
        self.assertGreater(len(self.X_train), 0, "X_train is empty")
        self.assertGreater(len(self.X_test), 0, "X_test is empty")
        self.assertGreater(len(self.y_train), 0, "y_train is empty")
        self.assertGreater(len(self.y_test), 0, "y_test is empty")

    # Test that X and y have matching lengths
    def test_shapes_align(self):
        self.assertEqual(
            len(self.X_train),
            len(self.y_train),
            "X_train and y_train have different lengths",
        )
        self.assertEqual(
            len(self.X_test),
            len(self.y_test),
            "X_test and y_test have different lengths",
        )

    # Test that train and test have the same features
    def test_feature_columns_match(self):
        self.assertListEqual(
            list(self.X_train.columns),
            list(self.X_test.columns),
            "Train and test feature columns do not match",
        )

    # Test that target variables are 1-dimensional
    def test_targets_are_1d(self):
        # y may be a Series or DataFrame; ensure 1D after conversion to Series
        y_train = pd.Series(self.y_train)
        y_test = pd.Series(self.y_test)
        self.assertEqual(y_train.ndim, 1)
        self.assertEqual(y_test.ndim, 1)

    # Test that target has no NaN values
    def test_no_missing_values_in_target(self):
        y_train = pd.Series(self.y_train)
        y_test = pd.Series(self.y_test)
        self.assertFalse(y_train.isna().any(), "y_train contains NaN values")
        self.assertFalse(y_test.isna().any(), "y_test contains NaN values")
        
if __name__ == "__main__":
    unittest.main()
