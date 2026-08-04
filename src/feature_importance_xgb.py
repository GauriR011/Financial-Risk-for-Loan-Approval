import joblib
import pandas as pd
from config import DEFAULT_MODEL_DIR

# 1. Load the pipeline
pipeline = joblib.load(DEFAULT_MODEL_DIR /"xgb_final_production_pipeline.joblib")

# 2. Extract the trained XGBoost estimator and the preprocessor
xgb_model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessing"]

# 3. Retrieve transformed feature names
# Handles one-hot encoded categories and created features accurately
feature_names = preprocessor.get_feature_names_out()

# 4. Extract feature importances (Gain is default for XGBoost)
importances = xgb_model.feature_importances_

# 5. Combine into a clean DataFrame
df_importance = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values(by="importance", ascending=False)
    .reset_index(drop=True)
)

print(df_importance)







# Output:
#                                feature  importance
# 0       num__TotalDebtToIncomeRatio_log    0.210273
# 1                     num__LoanDuration    0.063541
# 2                num__MonthlyIncome_log    0.055014
# 3                     num__InterestRate    0.048516
# 4                num__BankruptcyHistory    0.047483
# 5             num__PreviousLoanDefaults    0.044354
# 6       cat__EducationLevel_High School    0.034636
# 7                     num__NetWorth_log    0.034056
# 8      cat__EmploymentStatus_Unemployed    0.031433
# 9            num__LengthOfCreditHistory    0.028818
# 10           cat__EducationLevel_Master    0.027527
# 11                 num__TotalAssets_log    0.027069
# 12        cat__EducationLevel_Doctorate    0.026845
# 13                  num__LoanAmount_log    0.022427
# 14        cat__EducationLevel_Associate    0.021895
# 15        cat__HomeOwnershipStatus_Rent    0.020687
# 16                     num__CreditScore    0.017709
# 17       cat__HomeOwnershipStatus_Other    0.016748
# 18                             num__Age    0.012897
# 19                       num__Month_cos    0.012269
# 20           cat__MaritalStatus_Widowed    0.011782
# 21                       num__Month_sin    0.010216
# 22                  num__PaymentHistory    0.009844
# 23         cat__EducationLevel_Bachelor    0.009663
# 24         num__MonthlyDebtPayments_log    0.009508
# 25    cat__HomeOwnershipStatus_Mortgage    0.008555
# 26            num__TotalLiabilities_log    0.007296
# 27  cat__EmploymentStatus_Self-Employed    0.007241
# 28               num__DebtToIncomeRatio    0.006856
# 29                num__BaseInterestRate    0.006825
# 30         num__NumberOfCreditInquiries    0.006584
# 31            cat__MaritalStatus_Single    0.006543
# 32  cat__LoanPurpose_Debt Consolidation    0.006331
# 33          num__MonthlyLoanPayment_log    0.006211
# 34       cat__EmploymentStatus_Employed    0.006206
# 35                cat__LoanPurpose_Auto    0.005849
# 36           cat__LoanPurpose_Education    0.005818
# 37       num__CreditCardUtilizationRate    0.005720
# 38         num__NumberOfOpenCreditLines    0.005705
# 39       num__SavingsAccountBalance_log    0.005533
# 40         cat__HomeOwnershipStatus_Own    0.005504
# 41                       num__JobTenure    0.005375
# 42      num__CheckingAccountBalance_log    0.005250
# 43      num__UtilityBillsPaymentHistory    0.005248
# 44          cat__MaritalStatus_Divorced    0.005127
# 45                cat__LoanPurpose_Home    0.004900
# 46                 num__ApplicationYear    0.004899
# 47              num__NumberOfDependents    0.004161
# 48           cat__MaritalStatus_Married    0.003667
# 49               cat__LoanPurpose_Other    0.003390