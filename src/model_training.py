"""Entry point for model training"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import DATA_FILE, DEFAULT_MODEL_DIR, RANDOM_STATE
from .data import load_data, split_data
from .evaluation import evaluate_classifier, save_pipeline
from .pipeline import build_pipeline, tune_model


def get_models():
    models = {
        "lr" : LogisticRegression(max_iter=500, random_state = RANDOM_STATE),
        "rf" : RandomForestClassifier(n_estimators = 200, random_state=RANDOM_STATE, n_jobs=-1),
        "xgb" : XGBClassifier(objective = "binary:logistic", eval_metric = "logloss", random_state = RANDOM_STATE, n_jobs = -1)
    }

    return models

def main():
    # allows to pass data path, model, tuning directly from the command line terminal
    # a parser automatically packages the terminal command inputs into a clean Python object

    # initializing the argument parser
    parser = argparse.ArgumentParser(description="Train a loan approval model.")
    parser.add_argument("--data_path", type=Path, default=DATA_FILE)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", choices=["lr", "rf", "xgb"], default="rf")
    # choices=[...] restricts the user to choosing only one of these three exact models. The script will throw an error if anything else is typed.
    parser.add_argument("--tune", action="store_true", help="Tune the selected Random Forest model with CV.")
    # action="store_true" creates a boolean switch (flag). If you type --tune in the terminal, args.tune becomes True. If you leave it out, it defaults to False.

    # reading the inputs and saving them as a args object
    # You can now access them using dot notation (e.g., args.model).
    args = parser.parse_args()

    data = load_data(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    models = get_models()
    estimator = models[args.model]

    pipeline = build_pipeline(estimator, X_train)

    if args.tune:
        if args.model == "rf":
            # raise ValueError("--tune currently provides a search space for random_forest only.")
            parameter_space = {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            }
            search = tune_model(pipeline, parameter_space, X_train, y_train, n_iter=15, cv=5)
            pipeline = search.best_estimator_
            print("Best parameters:", search.best_params_)
            print(f"Best CV F1: {search.best_score_:.3f}")

        elif args.model == "lr":
            # raise ValueError("--tune currently provides a search space for random_forest only.")
            parameter_space = {
                "C": [0.1, 1, 10], # Controls how much you penalize the model for complexity. Smaller values, more penalty.
                "solver": ["liblinear", "lbfgs"],
                "class_weight": [None, "balanced"]
            }
            search = tune_model(pipeline, parameter_space, X_train, y_train, n_iter=15, cv=5)
            pipeline = search.best_estimator_
            print("Best parameters:", search.best_params_)
            print(f"Best CV F1: {search.best_score_:.3f}")
        else:
            parameter_space = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
            search = tune_model(pipeline, parameter_space, X_train, y_train, n_iter=15, cv=5)
            pipeline = search.best_estimator_
            print("Best parameters:", search.best_params_)
            print(f"Best CV F1: {search.best_score_:.3f}")
            

    else:
        pipeline.fit(X_train, y_train)


    evaluate_classifier(pipeline, X_val, y_val, "Validation")

    saved_path = save_pipeline(pipeline, args.model_path / f"{args.model}_trained_pipeline.joblib")
    print(f"Saved complete pipeline: {saved_path}")


if __name__ == "__main__":
    main()

# sample terminal command
# python models.py --model xgboost