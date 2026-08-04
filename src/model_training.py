"""Entry point for model training"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import DATA_FILE, DEFAULT_MODEL_DIR, RANDOM_STATE
from data import load_data, split_data
from evaluation import evaluate_classifier, save_pipeline
from pipeline import build_pipeline, tune_model


def get_models():
    models = {
        "lr" : LogisticRegression(class_weight= 'balanced', solver = 'liblinear', random_state = RANDOM_STATE),
        "rf" : RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "xgb" : XGBClassifier(objective = "binary:logistic", eval_metric = "logloss", random_state = RANDOM_STATE, n_jobs = -1)
    }

    return models

def main():
    # allows to pass data path, model, tuning directly from the command line terminal
    # a parser automatically packages the terminal command inputs into a clean Python object

    # initializing the argument parser
    print("Initialized Parser")
    parser = argparse.ArgumentParser(description="Train a loan approval model.")
    parser.add_argument("--data_path", type=Path, default=DATA_FILE)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", choices=["lr", "rf", "xgb"], default="rf", help="Model options : ['lr', 'rf', 'xgb']")
    # choices=[...] restricts the user to choosing only one of these three exact models. The script will throw an error if anything else is typed.
    parser.add_argument("--tune", choices=["grid", "random"], default=None, help="Tune the selected model with CV. Options : ['random', 'grid']")
    # action="store_true" creates a boolean switch (flag). If you type --tune in the terminal, args.tune becomes True. If you leave it out, it defaults to False.

    # reading the inputs and saving them as a args object
    # You can now access them using dot notation (e.g., args.model).
    args = parser.parse_args()

    print("Loading data")
    data = load_data(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    models = get_models()
    estimator = models[args.model]

    print("Building pipeline")
    pipeline = build_pipeline(estimator, X_train)

    if args.tune is not None:
        if args.model == "rf":
            # raise ValueError("--tune currently provides a search space for random_forest only.")
            parameter_space = {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            }

        elif args.model == "lr":
            # raise ValueError("--tune currently provides a search space for random_forest only.")
            parameter_space = {
                "model__C": [1, 10, 30], # Controls how much you penalize the model for complexity. Smaller values, more penalty.
                "model__solver": ["liblinear", "lbfgs"],
                "model__class_weight": [None, "balanced"]
            }

        else:
            parameter_space = {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }

        print("Hyperparameter tuning to select the best model")
        search = tune_model(pipeline, parameter_space, X_train, y_train, n_iter=15, cv=5, search_type = args.tune)
        pipeline = search.best_estimator_
        print("Best parameters:", search.best_params_)
        print(f"Best CV F1: {search.best_score_:.3f}")
            

    else:
        print("Training the model with default settings")
        pipeline.fit(X_train, y_train)

        # pipeline.named_steps["preprocessing"].set_output(transform="pandas")
        # X_train_preprocessed = pipeline[:-1].fit_transform(X_train, y_train)

        # print(f"Preprocessed shape: {X_train_preprocessed.shape}")
        # print(X_train_preprocessed.columns.tolist())

    print("Evaluating the model on Validation set")
    if args.tune is not None:
        evaluate_classifier(pipeline, X_val, y_val, model_name = f"{args.model}_best")
    else:
        evaluate_classifier(pipeline, X_val, y_val, model_name = f"{args.model}_base")

    print("Saving the model")

    if args.tune is not None:
        saved_path = save_pipeline(pipeline, args.model_path / f"{args.model}_best_model.joblib")
    else:
        saved_path = save_pipeline(pipeline, args.model_path / f"{args.model}_base_model.joblib")
    print(f"Saved complete pipeline: {saved_path}")


if __name__ == "__main__":
    main()

# sample terminal command
# python models.py --model lr