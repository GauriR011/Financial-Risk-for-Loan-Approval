"""Data loading and train/validation/test splitting."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TARGET_CLASSIFICATION, DATA_FILE

def load_data(path: Path = DATA_FILE, target:str = TARGET_CLASSIFICATION) -> pd.DataFrame:
    """Load raw data and confirm that the target exists."""

    data = pd.read_csv(path)
    if target not in data.columns:
       raise ValueError(f"Expected target column '{target}' was not found.")
    print("Data read successfully - data.py") 
    return data 


def split_data(
    data: pd.DataFrame,
    target: str = TARGET_CLASSIFICATION,
    holdout_size: float = 0.3,
    test_size: float = 0.5,
    random_state: int = RANDOM_STATE,
):
    """Return stratified raw train, validation, and test sets.

    All transformations are fitted later inside the pipeline, using training
    data only. This avoids leakage from validation or test rows.
    """

    print("Starting train test split... - data.py")
    if holdout_size <= 0 or test_size <= 0 or holdout_size> 0.5:
        raise ValueError("validation_size and test_size must be positive and the holdout size must be less than 50%.")

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_size,
        stratify=y_holdout,
        random_state=random_state,
    )
    print("Data split successfully - data.py")
    return X_train, X_val, X_test, y_train, y_val, y_test
