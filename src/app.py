"""Streamlit interface for a saved loan-approval prediction pipeline.

Run from the project root:
    streamlit run app.py
"""
from pathlib import Path
import re
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Allows models saved with `features.FeatureEngineering` to be loaded without
# retraining after the project was reorganised into a `src` package.
import features as legacy_features
from config import DATA_FILE, DEFAULT_MODEL_DIR, TARGET_CLASSIFICATION
from data import load_data

sys.modules.setdefault("features", legacy_features)


st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "xgb_final_production_pipeline.joblib"

# These are deliberately limited to fields that are likely to be meaningful
# for the first prediction. All remaining training columns stay optional and
# are imputed by the fitted pipeline when left blank.
REQUIRED_COLUMNS = [
    "MonthlyIncome", "MonthlyDebtPayments", "LoanAmount", "LoanDuration",
    "InterestRate", "BankruptcyHistory", "PreviousLoanDefaults", "EducationLevel",
    "EmploymentStatus", "TotalAssets", "TotalLiabilities", "HomeOwnershipStatus",
    "NumberOfDependents", "MaritalStatus", "LoanPurpose",
]

YES_NO_COLUMNS = {"BankruptcyHistory", "PreviousLoanDefaults"}
INTEGER_COLUMNS = {"LoanDuration", "NumberOfDependents"}

FIELD_DETAILS = {
    "MonthlyIncome": ("Monthly Income", "Please enter a numerical value without spaces, special characters, or separators."),
    "MonthlyDebtPayments": ("Monthly Debt Payment Amount", "Please enter a numerical value without spaces, special characters, or separators."),
    "LoanAmount": ("Loan Amount", "Please enter a numerical value without spaces, special characters, or separators."),
    "LoanDuration": ("Loan Duration", "Duration of loan (in months)."),
    "InterestRate": ("Interest Rate", "Annual interest rate. Enter a numerical value only, without a percent sign."),
    "BankruptcyHistory": ("Bankruptcy History", "Select Yes or No. The response is converted to 1 or 0 for the model."),
    "PreviousLoanDefaults": ("Previous Loan Defaults", "Select Yes or No. The response is converted to 1 or 0 for the model."),
    "EducationLevel": ("Education Level", "Select the applicant's highest education level."),
    "EmploymentStatus": ("Employment Status", "Select the applicant's current employment status."),
    "TotalAssets": ("Total Assets", "Please enter a numerical value without spaces, special characters, or separators."),
    "TotalLiabilities": ("Total Liabilities", "Please enter a numerical value without spaces, special characters, or separators."),
    "HomeOwnershipStatus": ("Home Ownership Status", "Select the applicant's current home ownership status."),
    "NumberOfDependents": ("Number Of Dependents", "Enter a whole numerical value."),
    "MaritalStatus": ("Marital Status", "Select the applicant's marital status."),
    "LoanPurpose": ("Loan Purpose", "Select the purpose of the loan application."),
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
    return joblib.load(model_path)


@st.cache_data
def load_reference_data(data_path: str) -> pd.DataFrame:
    """Load data only to build valid form choices and input-range checks."""
    return load_data(Path(data_path))


def label(column: str) -> str:
    """Turn a dataframe column name into a form label."""
    return FIELD_DETAILS.get(column, (" ".join(word.capitalize() for word in column.replace("_", " ").split()), ""))[0]


def is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series)


def field_key(column: str) -> str:
    return f"field_{column}"


def show_invalid_border(column: str) -> None:
    """Apply a red outline to a Streamlit widget after a failed submission."""
    if column not in st.session_state.get("invalid_fields", set()):
        return
    st.markdown(
        f"""<style>
        .st-key-{field_key(column)} input,
        .st-key-{field_key(column)} [data-baseweb="select"] > div {{
            border: 2px solid #d92727 !important;
            box-shadow: 0 0 0 1px #d92727 !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def make_input_widget(column: str, reference: pd.Series):
    """Render one required input and retain the raw value for validation."""
    field_label, help_text = FIELD_DETAILS[column]
    key = field_key(column)
    show_invalid_border(column)

    if column in YES_NO_COLUMNS:
        return st.selectbox(field_label, ["Select an option", "Yes", "No"], index=0, help=help_text, key=key)

    if is_categorical(reference):
        choices = sorted(reference.dropna().astype(str).unique().tolist())
        return st.selectbox(field_label, ["Select an option"] + choices, index=0, help=help_text, key=key)

    return st.text_input(field_label, value="", help=help_text, key=key)


def render_input_section(title: str, columns: list[str], reference_data: pd.DataFrame, values: dict):
    """Render a section in two columns and add values to the prediction record."""
    if not columns:
        return
    st.subheader(title)
    left, right = st.columns(2)
    for index, column in enumerate(columns):
        with left if index % 2 == 0 else right:
            values[column] = make_input_widget(column, reference_data[column])


def validate_and_convert_inputs(values: dict, reference_data: pd.DataFrame):
    """Return converted model inputs plus messages for missing/invalid fields."""
    errors, converted = {}, {}
    for column in REQUIRED_COLUMNS:
        value = values.get(column)
        if value in (None, "", "Select an option"):
            errors[column] = f"{label(column)} is required."
            continue

        if column in YES_NO_COLUMNS:
            converted[column] = 1 if value == "Yes" else 0
        elif is_categorical(reference_data[column]):
            converted[column] = value
        else:
            # Strip leading/trailing spaces
            clean_value = str(value).strip()
            
            # Standard single backslash inside raw strings
            pattern = r"^\d+$" if column in INTEGER_COLUMNS else r"^\d+(?:\.\d+)?$"
            
            if not re.fullmatch(pattern, clean_value):
                unit = "a whole number" if column in INTEGER_COLUMNS else "a numerical value without spaces, commas, or special characters"
                errors[column] = f"{label(column)} must be {unit}."
                continue
                
            converted[column] = int(clean_value) if column in INTEGER_COLUMNS else float(clean_value)
            
    return converted, errors


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


# def readable_feature_name(name: str) -> str:
#     """Make transformer feature names understandable without an LLM."""
#     name = name.replace("numeric__", "").replace("categorical__", "")
#     return label(name.replace("_", " "))

# def readable_feature_name(name: str) -> str:
#     """Make transformer feature names understandable without an LLM."""
#     # Strip pipeline prefixes generated by ColumnTransformer
#     clean_name = name.split("__")[-1]
    
#     # Strip log suffixes if present
#     clean_name = clean_name.replace("_log", "")
    
#     # Capitalize and format nicely
#     return label(clean_name.replace("_", " "))


# # def get_shap_contributions(pipeline, raw_row: pd.DataFrame) -> pd.DataFrame:
# #     """Calculate local SHAP values for an XGBoost-like tree model pipeline."""
# #     engineered = pipeline.named_steps["feature_engineering"].transform(raw_row)
# #     preprocessor = pipeline.named_steps["preprocessing"]
# #     transformed = preprocessor.transform(engineered)
# #     transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
# #     feature_names = preprocessor.get_feature_names_out()

# #     estimator = pipeline.named_steps["model"]
# #     explainer = shap.TreeExplainer(estimator)
# #     shap_values = explainer.shap_values(transformed_dense)
# #     if isinstance(shap_values, list):
# #         shap_values = shap_values[1]  # Positive class: loan approval.
# #     values = np.asarray(shap_values)[0]

# #     return pd.DataFrame({
# #         "feature": [readable_feature_name(name) for name in feature_names],
# #         "shap_value": values,
# #     }).sort_values("shap_value", key=np.abs, ascending=False)

# def get_shap_contributions(pipeline, raw_row: pd.DataFrame) -> pd.DataFrame:
#     """Calculate local SHAP values for an XGBoost-like tree model pipeline."""
#     engineered = pipeline.named_steps["feature_engineering"].transform(raw_row)
#     preprocessor = pipeline.named_steps["preprocessing"]
#     transformed = preprocessor.transform(engineered)
    
#     # Handle sparse matrices and force float dtype
#     if hasattr(transformed, "toarray"):
#         transformed_dense = transformed.toarray().astype(np.float64)
#     else:
#         transformed_dense = np.asarray(transformed, dtype=np.float64)

#     feature_names = preprocessor.get_feature_names_out()

#     estimator = pipeline.named_steps["model"]
    
#     # Use TreeExplainer for XGBoost / LightGBM
#     explainer = shap.TreeExplainer(estimator)
#     shap_values = explainer.shap_values(transformed_dense)

#     # Handle multi-class / list output format safely
#     if isinstance(shap_values, list):
#         shap_values = shap_values[1]  # Positive class: loan approval
#     elif len(shap_values.shape) == 3:
#         shap_values = shap_values[:, :, 1]

#     values = np.asarray(shap_values).ravel()

#     return pd.DataFrame({
#         "feature": [readable_feature_name(name) for name in feature_names],
#         "shap_value": values,
#     }).sort_values("shap_value", key=np.abs, ascending=False)

def readable_feature_name(name: str) -> str:
    """Format encoded column names into user-friendly labels."""
    # Remove prefix (cat__ or num__)
    prefix, _, raw_name = name.partition("__")
    
    if prefix == "cat":
        # Handles OneHotEncoder outputs like 'EmploymentStatus_Self-Employed'
        if "_" in raw_name:
            col_name, val = raw_name.split("_", 1)
            return f"{label(col_name)}: {val}"
        return label(raw_name)
    
    # Clean up numeric features like 'MonthlyIncome_log'
    clean_name = raw_name.replace("_log", "")
    return label(clean_name)


def get_shap_contributions(
    pipeline,
    raw_row: pd.DataFrame,
    reference_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a local, model-agnostic SHAP explanation.

    This avoids shap.TreeExplainer, which currently has compatibility
    issues with XGBoost 3.x models.
    """

    raw_features = raw_row.columns.tolist()

    # Use a small real-data background sample to make the explanation feasible.
    background = (
        reference_data[raw_features]
        .sample(n=min(20, len(reference_data)), random_state=42)
        .copy()
    )

    def predict_approval_probability(data):
        # SHAP may pass a NumPy array, so restore DataFrame column names.
        input_df = pd.DataFrame(data, columns=raw_features)
        return pipeline.predict_proba(input_df)[:, 1]

    explainer = shap.KernelExplainer(
        predict_approval_probability,
        background
    )

    shap_values = explainer.shap_values(
        raw_row,
        nsamples=200
    )

    values = np.asarray(shap_values).ravel()

    return pd.DataFrame({
        "feature": [label(column) for column in raw_features],
        "shap_value": values,
    }).sort_values("shap_value", key=np.abs, ascending=False)


def render_explanation(contributions: pd.DataFrame):
    """Show SHAP values and a deterministic plain-language summary."""
    top = contributions.head(8).sort_values("shap_value")
    # fig, ax = plt.subplots(figsize=(8, 4.5))
    fig, ax = plt.subplots(figsize=(6, 3.5))
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

    # with st.sidebar:
        # st.header("Model settings")
        # model_path = st.text_input("Production model path", str(DEFAULT_MODEL_PATH))
        # st.caption("Use the final production pipeline created after test evaluation.")
    try:
        model_path = DEFAULT_MODEL_PATH
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
            available = [column for column in REQUIRED_COLUMNS if column in section_columns and column in raw_features]
            assigned.update(available)
            render_input_section(section, available, reference_data, values)

        submitted = st.form_submit_button("Get prediction", type="primary")

    if not submitted:
        return

    converted_values, errors = validate_and_convert_inputs(values, reference_data)
    st.session_state["invalid_fields"] = set(errors)
    if errors:
        # The widgets have already been rendered in this run, so apply their
        # error styles now as well as on the next submission attempt.
        for column in errors:
            show_invalid_border(column)
        st.error("Please correct the highlighted fields before requesting a prediction.")
        for message in errors.values():
            st.error(message)
        st.stop()

    # Preserve every raw training column and its original order. The fitted
    # pipeline—not Streamlit—performs engineering, imputation, encoding, and scaling.
    # applicant = pd.DataFrame([{column: converted_values.get(column) for column in raw_features}])
    
    applicant = pd.DataFrame([{column: converted_values.get(column) for column in raw_features}])

    # Force convert numerical columns to proper numeric types
    for col in applicant.columns:
        if col in reference_data.columns and pd.api.types.is_numeric_dtype(reference_data[col]):
            applicant[col] = pd.to_numeric(applicant[col], errors='coerce')


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
            contributions = get_shap_contributions(
                pipeline,
                applicant,
                reference_data
            )
            render_explanation(contributions)
        except Exception as error:
            st.warning(
                "A SHAP explanation could not be generated. Confirm that the deployed model is an XGBoost tree model and that `shap` is installed."
            )
            st.caption(f"Technical detail: {error}")


if __name__ == "__main__":
    main()
