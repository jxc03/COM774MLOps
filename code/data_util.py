import pandas as pd
from pathlib import Path

def load_train_test(data_dir: str):
    data_dir = Path(data_dir)
    
    X_train = pd.read_csv(data_dir / "X_train_classification.csv")
    X_test  = pd.read_csv(data_dir / "X_test_classification.csv")
    
    y_train_raw = pd.read_csv(data_dir / "y_train_classification.csv")["priority_encoded"]
    y_test_raw  = pd.read_csv(data_dir / "y_test_classification.csv")["priority_encoded"]
    
    
    # To fix error: If y_ CSVs have only one column, this will grab it as a Series
    #if "priority_encoded" in y_train_df.columns:
        #y_train = y_train_df["priority_encoded"]
        #y_test = y_test_df["priority_encoded"]
    #else:
        # fall back to "first column" if the column name is different
        #y_train = y_train_df.iloc[:, 0]
        #y_test = y_test_df.iloc[:, 0]

    # To fix error: AttributeError: 'Series' object has no attribute 'columns'
    # Handle y as either Series or DataFrame
    def extract_target(y_raw: pd.DataFrame | pd.Series):
        # If it's already a Series, just return it
        if isinstance(y_raw, pd.Series):
            return y_raw

        # If it's a DataFrame, pick the right column
        if isinstance(y_raw, pd.DataFrame):
            # Prefer a named column if present
            if "priority_encoded" in y_raw.columns:
                return y_raw["priority_encoded"]
            else:
                # Otherwise just take the first column
                return y_raw.iloc[:, 0]

        # Fallback: convert to Series
        return pd.Series(y_raw)

    y_train = extract_target(y_train_raw)
    y_test  = extract_target(y_test_raw)

    return X_train, X_test, y_train, y_test
