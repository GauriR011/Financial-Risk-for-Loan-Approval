"""Pipeline construction and hyperparameter tuning."""
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from config import LEAKAGE_OR_UNUSED_COLUMNS, RANDOM_STATE
from features import FeatureEngineering, scale_and_encode

def build_pipeline(model, X_train: pd.DataFrame, drop_cols = LEAKAGE_OR_UNUSED_COLUMNS) -> Pipeline:
    """Create one end-to-end, leakage-safe scikit-learn pipeline."""    
    return Pipeline([
        ("feature_engineering", FeatureEngineering(drop_additional_cols=drop_cols)),
        # ("preprocessing", scale_and_encode(X_train, drop_cols)),
        ("preprocessing", scale_and_encode()),
        ("model", model),
    ])


# there are 2 modes : "grid" and "random"
def tune_model(
        pipeline: Pipeline, 
        parameter_space, 
        X_train, y_train, 
        search_type = "random", 
        n_iter = 20, 
        cv = 5,
        scoring = "f1"):
    """Tune the whole pipeline using training folds only."""
    print("Tuning model - pipeline.py")
    if search_type == "grid":
        search = GridSearchCV(pipeline, parameter_space, cv = cv, scoring=scoring, n_jobs = -1)
    else:
        search = RandomizedSearchCV(pipeline, parameter_space, cv=cv, n_iter=n_iter, scoring=scoring, random_state=RANDOM_STATE, n_jobs=-1) 
    print("Tuning successful - pipeline.py")
    return search.fit(X_train, y_train)