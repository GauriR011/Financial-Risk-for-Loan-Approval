"""Feature engineering and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Create features from exisiting ones, applying log transformations"""

    log_transform_columns = [
        'SavingsAccountBalance', 'TotalLiabilities', 'TotalAssets', 
        'CheckingAccountBalance', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 
        'AnnualIncome', 'MonthlyIncome', 'LoanAmount', 'MonthlyDebtPayments', 'NetWorth'
    ]

    columns_to_remove = [
        "ApplicationDate", "AnnualIncome", "Experience", "TotalAssets",
        "TotalLiabilities", "Monthly_LoanToIncomeRatio", "MonthlyDebtPayments",
        "LoanAmount", "MonthlyIncome", "SavingsAccountBalance",
        "CheckingAccountBalance", "MonthlyLoanPayment", "TotalDebtToIncomeRatio", "NetWorth"
    ]

    def __init__(self, drop_additional_cols = None):
        # defining hyperparameters
        self.drop_additional_cols = drop_additional_cols

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _correct_application_year(years: pd.Series) -> pd.Series:
        """Map anomalously encoded years to the intended 2018–2024 range."""

        corrected_years = years.copy() 
        mappings= [
            (2025, 2031, 2018), (2032, 2038, 2019), (2039, 2045, 2020),
            (2046, 2052, 2021), (2053, 2059, 2022), (2060, 2066, 2023)
        ]

        for lower, upper, replacement in mappings:
            corrected_years = corrected_years.mask(corrected_years.between(lower, upper), replacement)
        corrected_years = corrected_years.mask(corrected_years >= 2067, 2024)

    def transform(self, X: pd.DataFrame):

        df = X.copy()

        # Creating new features
        if {'MonthlyDebtPayments', 'MonthlyIncome'}.issubset(df.columns):
            income = df["MonthlyIncome"].replace(0, np.nan)
            df['DebtToIncomeRatio'] = df['MonthlyDebtPayments'] / income

        if {"MonthlyLoanPayment", "MonthlyIncome"}.issubset(df.columns):
            income = df["MonthlyIncome"].replace(0, np.nan)
            df["Monthly_LoanToIncomeRatio"] = df["MonthlyLoanPayment"] / income

        # Extracting Application Month and Year from Application Date
        if "ApplicationDate" in df.columns:
            dates = pd.to_datetime(df["ApplicationDate"], errors="coerce")
            df["ApplicationYear"] = self._correct_application_year(dates.dt.year)

            month = dates.dt.month
            df["Month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
            df["Month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

        # Apply log transformation
        for column in self.log_transform_columns:
            if column in df.columns: 
                df[f"{column}_log"] = np.log1p(df[column].clip(lower=0))

        total_columns_to_drop = set(self.columns_to_remove) | set(self.drop_columns or [])
        df.drop(columns = list(total_columns_to_drop), inplace = True, errors = "ignore")

        return df


def extract_feature_groups(X_train: pd.DataFrame, drop_columns=None):
    """Determine numeric and categorical columns of the dataset"""

    # apply transformations to the training data
    tranformations = FeatureEngineering(drop_additional_cols = drop_columns)
    all_columns = tranformations.fit_transform(X_train)

    # extract the categorical and numerical columns from the transformed data
    categorical_cols = all_columns.select_dtypes(include = ["object", "category", "bool"]).columns.tolist()
    numerical_cols = all_columns.select_dtypes(include = np.number).columns.tolist()

    return categorical_cols,  numerical_cols



def scale_and_encode(X_train: pd.DataFrame, drop_columns = None) -> ColumnTransformer:
    """Build a pipeline with imputation, scaling, and one-hot encoding steps."""

    categorical_cols, numerical_cols = extract_feature_groups(X_train, drop_columns=drop_columns)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_cols),
        ("numeric", numerical_pipeline, numerical_cols),
    ], remainder="drop")
