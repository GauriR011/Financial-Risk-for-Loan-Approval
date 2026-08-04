"""Feature engineering and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
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
        "CheckingAccountBalance", "MonthlyLoanPayment", "TotalDebtToIncomeRatio",
        'AnnualIncome_log'
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
        print("Application year modified successfully - features.py")
        return corrected_years

    def transform(self, X: pd.DataFrame):
        print("Starting data transformation... - features.py")
        df = X.copy()

        #Fill completely missing/unpassed log columns with NaN so operations don't fail
        for column in self.log_transform_columns:
            if column not in df.columns:
                df[column] = np.nan

        # Creating new features

        # Monthly Loan Payment
        if {"InterestRate", "LoanDuration", "LoanAmount"}.issubset(df.columns):
            # 1. Convert annual InterestRate to monthly rate
            r = df["InterestRate"] / 12

            # 2. Extract duration in months and principal
            n = df["LoanDuration"]
            P = df["LoanAmount"]

            # 3. Calculate MonthlyLoanPayment using the amortization formula
            # M = P * [r(1 + r)^n] / [(1 + r)^n - 1]
            df["MonthlyLoanPayment"] = (P * (r * (1 + r) ** n) / (((1 + r) ** n) - 1))


        # DebtToIncomeRatio
        if {'MonthlyDebtPayments', 'MonthlyIncome'}.issubset(df.columns):
            income = df["MonthlyIncome"].replace(0, np.nan)
            df['DebtToIncomeRatio'] = df['MonthlyDebtPayments'] / income

        # Monthly_LoanToIncomeRatio
        if {"MonthlyIncome"}.issubset(df.columns):
            income = df["MonthlyIncome"].replace(0, np.nan)
            df["Monthly_LoanToIncomeRatio"] = df["MonthlyLoanPayment"] / income

        # TotalDebtToIncomeRatio
        if {"MonthlyIncome", "MonthlyDebtPayments"}.issubset(df.columns):
            income = df["MonthlyIncome"].replace(0, np.nan)
            df["TotalDebtToIncomeRatio"] = (df["MonthlyDebtPayments"] + df["MonthlyLoanPayment"]) / income

        if{"TotalAssets", "TotalLiabilities"}.issubset(df.columns):
            df['NetWorth'] = df["TotalAssets"] - df["TotalLiabilities"]

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
                # Convert to numeric, replace non-numeric/None with NaN
                numeric_series = pd.to_numeric(df[column], errors='coerce')
                
                # Impute missing values with 0 temporarily for log transform (or median if needed)
                # clip(lower=0) prevents negative numbers from breaking log1p
                imputed_series = numeric_series.fillna(0).clip(lower=0)
                
                # Assign log transformed values back
                df[f"{column}_log"] = np.log1p(imputed_series)
                
                # Re-insert NaN where original data was missing so SimpleImputer can handle it later
                df.loc[numeric_series.isna(), f"{column}_log"] = np.nan

        # dropping columns
        total_columns_to_drop = set(self.columns_to_remove) | set(self.drop_additional_cols or [])
        df.drop(columns = list(total_columns_to_drop), inplace = True, errors = "ignore")

        print("Data transformation successful - features.py")
        return df

def scale_and_encode() -> ColumnTransformer:
    """Builds the encoding and scaling step dynamically without needing X_train."""
    print("Starting scaling and encoding... - features.py")

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    print("Scaling and encoding successful - features.py")

    # make_column_selector inspects the incoming DataFrame dynamically during pipeline.fit()
    return ColumnTransformer([
        ("cat", categorical_pipeline, make_column_selector(dtype_include=["object", "category", "bool"])),
        ("num", numerical_pipeline, make_column_selector(dtype_include=np.number)),
    ], remainder="drop")
