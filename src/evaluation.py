"""Model evaluation, plots, and serialisation."""

from pathlib import Path

from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import(
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score
)

def evaluate_classifier(model : Pipeline, X, y, dataset_name = "Val", show_plot = True):
    """Calculate and print classification metrics for a fitted pipeline."""

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc" : roc_auc_score(y, y_prob),
        "average_precision" : average_precision_score(y, y_prob)
    }

    print(f"\\n{dataset_name} classification report")
    print(classification_report(y, y_pred, digits=3))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f} | Average precision: {metrics['average_precision']:.3f}")
    if show_plot:
        ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap="Blues")
        plt.title(f"{dataset_name} confusion matrix")
        plt.show()

    return metrics


def compare_roc_pr_curves(fitted_models, X, y):
    """Plot ROC and precision–recall curves for several fitted pipelines."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, model in fitted_models.items():
        probabilities = model.predict_proba(X)[:, 1]
        RocCurveDisplay.from_predictions(y, probabilities, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y, probabilities, name=name, ax=axes[1])
    axes[0].set_title("ROC curves")
    axes[1].set_title("Precision–recall curves")
    plt.tight_layout()
    plt.show()    


def save_pipeline(model, output_path: Path) -> Path:
    """Save the complete pipeline so inference uses identical transformations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path