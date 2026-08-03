import pandas as pd
import joblib
import argparse
from pathlib import Path
from .config import DATA_FILE, DEFAULT_MODEL_DIR
from .data import load_data, split_data
from .evaluation import evaluate_classifier

def main():
    parser = argparse.ArgumentParser(description="Evaluating a trained model.")
    parser.add_argument("--data_path", type=Path, default=DATA_FILE)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model", choices=["lr", "rf", "xgb"], default="rf")

    # reading the inputs and saving them as a args object
    # You can now access them using dot notation (e.g., args.model).
    args = parser.parse_args()


    # Loading the original data splits
    data = load_data(DATA_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)


    # Loading the final model
    model_path = DEFAULT_MODEL_DIR / f"{args.model}_trained_pipeline.joblib"
    try:
        final_pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: The file does not exist. Please check whether the model has been trained first.")
    except Exception as e:
        # Runs for any other unexpected errors, saving the error message to 'e'
        print(f"{e}")

    # Refitting after model selection
    X = pd.concat([X_train, X_val])
    y = pd.concat([y_train, y_val])

    final_pipeline.fit(X, y)

    evaluate_classifier(final_pipeline, X_test, y_test, "Test")

    # Saving the final production model for deployment
    production_path = DEFAULT_MODEL_DIR / f"{model_name}_final_production_pipeline.joblib"
    saved_path = save_pipeline(pipeline, production_path)
    print(f"Saved production-ready pipeline: {saved_path}")

if __name__ == "__main__":
    main()
