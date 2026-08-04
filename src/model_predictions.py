import pandas as pd
import joblib
import argparse
from pathlib import Path
from evaluation import save_pipeline, evaluate_classifier
from config import DATA_FILE, DEFAULT_MODEL_DIR
from data import load_data, split_data

def main():
    parser = argparse.ArgumentParser(description="Evaluating a trained model.")
    parser.add_argument("--data_path", type=Path, default=DATA_FILE)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", choices=["lr", "rf", "xgb"], default="rf")
    parser.add_argument("--tune", action="store_true", help="Do you want to evaluate the tuned or base model?")
    # reading the inputs and saving them as a args object
    # You can now access them using dot notation (e.g., args.model).
    args = parser.parse_args()


    # Loading the original data splits
    data = load_data(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)


    # Loading the final model
    if args.tune:
        full_model_path = args.model_path / f"{args.model}_best_model.joblib"
    else:
        full_model_path = args.model_path / f"{args.model}_base_model.joblib"

    try:
        final_pipeline = joblib.load(full_model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{full_model_path}' does not exist. Please train the model first.")
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return

    # Refitting after model selection
    X = pd.concat([X_train, X_val])
    y = pd.concat([y_train, y_val])

    final_pipeline.fit(X, y)
    if args.tune:
        evaluate_classifier(final_pipeline, X_test, y_test, dataset_name = "Test", model_name = f"{args.model}_best")
    else:
        evaluate_classifier(final_pipeline, X_test, y_test, dataset_name = "Test", model_name = f"{args.model}_base")

    # Saving the final production model for deployment
    production_path = args.model_path / f"{args.model}_final_production_pipeline.joblib"
    saved_path = save_pipeline(final_pipeline, production_path)
    print(f"Saved production-ready pipeline: {saved_path}")

if __name__ == "__main__":
    main()
