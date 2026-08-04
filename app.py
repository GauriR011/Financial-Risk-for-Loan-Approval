"""Streamlit interface for a saved loan-approval prediction pipeline.

Run from the project root:
    streamlit run loan_approval_streamlit_dashboard/app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import DATA_FILE, DEFAULT_MODEL_DIR, TARGET_CLASSIFICATION
from src.data import load_data


st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xgb_final_production_pipeline.joblib"

# These are deliberately limited to fields that are likely to be meaningful
# for the first prediction. All remaining training columns stay optional and
# are imputed by the fitted pipeline when left blank.
REQUIRED_COLUMNS = {
    "Age", "MonthlyIncome", "MonthlyDebtPayments", "CreditScore", "LoanAmount",
    "LoanDuration", "InterestRate", "TotalDebtToIncomeRatio", "PaymentHistory",
    "LengthOfCreditHistory", "EmploymentStatus", "LoanPurpose",
}

FIELD_GROUPS = {
    "Personal profile": {
        "Age", "EducationLevel", "MaritalStatus", "NumberOfDependents"
    },
    "Employment and income": {
        "EmploymentStatus", "Experience", "JobTenure", "MonthlyIncome"
    },
    "Credit profile": {
        "CreditScore", "LengthOfCreditHistory", "NumberOfOpenCreditLines",
        "NumberOfCreditInquiries", "CreditCardUtilizationRate", "PaymentHistory",
        "UtilityBillsPaymentHistory", "PreviousLoanDefaults", "BankruptcyHistory",
    },
    "Financial position": {
        "MonthlyDebtPayments", "TotalDebtToIncomeRatio", "SavingsAccountBalance",
        "CheckingAccountBalance", "TotalAssets", "TotalLiabilities", "NetWorth",
    },
    "Loan request": {
        "LoanAmount", "LoanDuration", "LoanPurpose", "InterestRate", "BaseInterestRate",
        "MonthlyLoanPayment", "HomeOwnershipStatus",
    },
    "Application details": {"ApplicationDate"},
}


@st.cache_resource
def load_pipeline(model_path: str):
    """Load once per Streamlit session, rather than on every widget update."""
    return Path(joblib.load(model_path))


@st.cache_data
def load_reference_data(data_path: str) -> pd.DataFrame:
    """Load data only to build valid form choices and input-range checks."""
    return load_data(Path(data_path))


def label(column: str) -> str:
    """Turn a dataframe column name into a form label."""
    return " ".join(word.capitalize() for word in column.replace("_", " ").split())


def is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series)


def make_input_widget(column: str, reference: pd.Series, required: bool):
    """Return a raw input value. Optional blank fields are returned as None."""
    field_label = f"{label(column)}{' *' if required else ' (optional)'}"

    if column == "ApplicationDate":
        value = st.date_input(field_label, value=None, help="Optional; the pipeline derives time features when supplied.")
        return value.isoformat() if value else None

    if is_categorical(reference):
        choices = sorted(reference.dropna().astype(str).unique().tolist())
        if required:
            return st.selectbox(field_label, choices)
        return st.selectbox(field_label, ["— Not provided —"] + choices, index=0)

    median = float(reference.median()) if reference.notna().any() else 0.0
    if required:
        return st.number_input(field_label, value=median)

    raw_value = st.text_input(field_label, value="", help="Leave blank to let the pipeline impute this value.")
    if raw_value.strip() == "":
        return None
    try:
        return float(raw_value)
    except ValueError:
        st.error(f"{label(column)} must be a number.")
        return None


def render_input_section(title: str, columns: list[str], reference_data: pd.DataFrame, values: dict):
    """Render a section in two columns and add values to the prediction record."""
    if not columns:
        return
    st.subheader(title)
    left, right = st.columns(2)
    for index, column in enumerate(columns):
        with left if index % 2 == 0 else right:
            value = make_input_widget(column, reference_data[column], column in REQUIRED_COLUMNS)
            values[column] = None if value == "— Not provided —" else value


def check_input_ranges(row: pd.DataFrame, reference_data: pd.DataFrame) -> list[str]:
    """Warn when provided numeric values are outside the observed training range."""
    warnings = []
    for column in row.columns:
        value = row.at[0, column]
        if pd.isna(value) or not pd.api.types.is_numeric_dtype(reference_data[column]):
            continue
        minimum, maximum = reference_data[column].min(), reference_data[column].max()
        if value < minimum or value > maximum:
            warnings.append(f"{label(column)} is outside the range observed during training ({minimum:,.2f}–{maximum:,.2f}).")
    return warnings


def readable_feature_name(name: str) -> str:
    """Make transformer feature names understandable without an LLM."""
    name = name.replace("numeric__", "").replace("categorical__", "")
    return label(name.replace("_", " "))


def get_shap_contributions(pipeline, raw_row: pd.DataFrame) -> pd.DataFrame:
    """Calculate local SHAP values for an XGBoost-like tree model pipeline."""
    engineered = pipeline.named_steps["feature_engineering"].transform(raw_row)
    preprocessor = pipeline.named_steps["preprocessing"]
    transformed = preprocessor.transform(engineered)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    feature_names = preprocessor.get_feature_names_out()

    estimator = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed_dense)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class: loan approval.
    values = np.asarray(shap_values)[0]

    return pd.DataFrame({
        "feature": [readable_feature_name(name) for name in feature_names],
        "shap_value": values,
    }).sort_values("shap_value", key=np.abs, ascending=False)


def render_explanation(contributions: pd.DataFrame):
    """Show SHAP values and a deterministic plain-language summary."""
    top = contributions.head(8).sort_values("shap_value")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = np.where(top["shap_value"] >= 0, "#2E8B57", "#C0392B")
    ax.barh(top["feature"], top["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution toward approval probability")
    ax.set_title("Largest factors affecting this prediction")
    st.pyplot(fig, clear_figure=True)

    positive = contributions[contributions["shap_value"] > 0].head(3)["feature"].tolist()
    negative = contributions[contributions["shap_value"] < 0].head(3)["feature"].tolist()
    positive_text = ", ".join(positive) if positive else "No individual factor increased the estimate"
    negative_text = ", ".join(negative) if negative else "No individual factor reduced the estimate"
    st.write(f"**Factors increasing the estimated approval likelihood:** {positive_text}.")
    st.write(f"**Factors reducing the estimated approval likelihood:** {negative_text}.")


def main():
    st.title("🏦 Loan Approval Predictor")
    st.caption("An educational model demonstration using a saved machine-learning pipeline.")

    st.info(
        "**How to use this tool:** Complete the required fields marked with *; add optional details if available; "
        "then select **Get prediction**. The application returns a predicted outcome, estimated probability, and a local SHAP explanation."
    )
    st.warning(
        "This is not a real lending decision or financial advice. A model prediction should never be the sole basis for a loan decision."
    )

    with st.sidebar:
        st.header("Model settings")
        model_path = st.text_input("Production model path", str(DEFAULT_MODEL_PATH))
        st.caption("Use the final production pipeline created after test evaluation.")

    try:
        reference_data = load_reference_data(str(DATA_FILE))
        pipeline = load_pipeline(model_path)
    except FileNotFoundError as error:
        st.error(f"Could not load a required file: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Could not initialise the application: {error}")
        st.stop()

    raw_features = [column for column in reference_data.columns if column != TARGET_CLASSIFICATION]
    values = {}

    with st.form("loan_application"):
        assigned = set()
        for section, section_columns in FIELD_GROUPS.items():
            available = [column for column in raw_features if column in section_columns]
            assigned.update(available)
            render_input_section(section, available, reference_data, values)

        extra_columns = [column for column in raw_features if column not in assigned]
        with st.expander("Additional optional information"):
            render_input_section("Additional details", extra_columns, reference_data, values)

        submitted = st.form_submit_button("Get prediction", type="primary")

    if not submitted:
        return

    # Preserve every raw training column and its original order. The fitted
    # pipeline—not Streamlit—performs engineering, imputation, encoding, and scaling.
    applicant = pd.DataFrame([{column: values.get(column) for column in raw_features}])
    for warning in check_input_ranges(applicant, reference_data):
        st.warning(warning)

    try:
        approval_probability = float(pipeline.predict_proba(applicant)[0, 1])
        prediction = int(pipeline.predict(applicant)[0])
    except Exception as error:
        st.error(f"The model could not score this application: {error}")
        st.stop()

    outcome = "Approved" if prediction == 1 else "Rejected"
    st.divider()
    st.header("Prediction result")
    left, middle, right = st.columns(3)
    left.metric("Predicted outcome", outcome)
    middle.metric("Estimated approval probability", f"{approval_probability:.1%}")
    right.metric("Estimated rejection probability", f"{1 - approval_probability:.1%}")
    st.caption(
        "These are model-estimated probabilities based on historical training patterns, not a guarantee or a calibrated confidence statement."
    )

    with st.expander("Why did the model make this prediction?", expanded=True):
        st.write(
            "The model begins with its learned baseline and adjusts the estimate using the applicant information. "
            "Green contributions push the estimate toward approval; red contributions push it toward rejection."
        )
        try:
            contributions = get_shap_contributions(pipeline, applicant)
            render_explanation(contributions)
        except Exception as error:
            st.warning(
                "A SHAP explanation could not be generated. Confirm that the deployed model is an XGBoost tree model and that `shap` is installed."
            )
            st.caption(f"Technical detail: {error}")


if __name__ == "__main__":
    main()
