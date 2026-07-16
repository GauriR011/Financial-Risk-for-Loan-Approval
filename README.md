# Financial-Risk-for-Loan-Approval

## Business Problem: Optimizing Loan Portfolio Profitability While Minimizing Risk Exposure
**Background:**    
A financial institution wants to grow its loan portfolio by approving more applications without significantly increasing its default risk. Historically, overly cautious approval strategies have led to lost revenue opportunities, while lenient strategies have increased default rates and write-offs.

**Objective:**    
Use predictive modeling to identify applicants who are both likely to repay their loans and profitable to approve, balancing approval rates with financial risk and expected returns.

### Key Question to Address:
<!-- ##### Risk Assessment (Regression Task)

1) What is the predicted RiskScore for a new applicant, indicating the likelihood of default?

2) How does this score correlate with key variables like credit history, debt-to-income ratio, or loan purpose? -->

**Loan Approval (Classification Task)** - Should this applicant be approved or denied for a loan based on their financial profile?

<!-- ##### Strategic Decisioning

1) How can we segment applicants (e.g., low-risk/high-value, high-risk/low-value) to refine lending strategies?

2) What threshold on the RiskScore corresponds to optimal approval criteria (balancing profit vs. risk)?

3) Can we dynamically adjust interest rates based on predicted risk and expected loss? -->


## Dataset
The dataset includes the following columns:
| Column Name | Column Description |
| ------------- | ------------- |
| ApplicationDate | Loan application date |
| Age | Applicant's age |
| AnnualIncome | Yearly income |
| CreditScore | Creditworthiness score |
| EmploymentStatus | Job situation |
| EducationLevel | Highest education attained |
| Experience | Work experience |
| LoanAmount | Requested loan size |
| LoanDuration | Loan repayment period |
| MaritalStatus | Applicant's marital state |
| NumberOfDependents | Number of dependents |
| HomeOwnershipStatus | Homeownership type |
| MonthlyDebtPayments | Monthly debt obligations |
| CreditCardUtilizationRate | Credit card usage percentage |
| NumberOfOpenCreditLines | Active credit lines |
| NumberOfCreditInquiries | Credit checks count |
| DebtToIncomeRatio | Debt to income proportion |
| BankruptcyHistory | Bankruptcy records |
| LoanPurpose | Reason for loan |
| PreviousLoanDefaults | Prior loan defaults |
| PaymentHistory | Past payment behavior |
| LengthOfCreditHistory | Credit history duration |
| SavingsAccountBalance | Savings account amount |
| CheckingAccountBalance | Checking account funds |
| TotalAssets | Total owned assets |
| TotalLiabilities | Total owed debts |
| MonthlyIncome | Income per month |
| UtilityBillsPaymentHistory | Utility payment record |
| JobTenure | Job duration |
| NetWorth | Total financial worth |
| BaseInterestRate | Starting interest rate |
| InterestRate | Applied interest rate |
| MonthlyLoanPayment | Monthly loan payment | 
| TotalDebtToIncomeRatio | Total debt against income | 
| LoanApproved | Loan approval status | 
| RiskScore | Risk assessment score | 


## Implementation
1) Understanding the Dataset (performing inintal analysis)
    - Overall shape of the dataset (20000,36)
    - Presence of any null values (no null values)
    - Checking the datatypes of the columns (float, int and object)

2) Data Cleaning
    - Dropping unecessary columns 
    - Cleaning the Application Date column (extracting the month and year from the date)
    - Truncating the floating point values (rounding to 3 decimal places)
    - Checking for consistency in column data

3) Feature Correlation
    - Checking how much each feature contributed to determining the target column.
    - We are creating a correlation matrix with the target variable and then sorting the features by the magnitude of correlation.
    - We are using a heatmap color code the features based on the magnitude.
    - **Observation**: The following columns play a significant role in determining whether the loan is approved or rejected
        -  (having positive correlation) 'MonthlyIncome_log', 'NetWorth', 'CreditScore', 'Age', 'EducationLevel_Master', 'LengthOfCreditHistory'
        -  (having negative correlation) 'DebtToIncomeRatio', 'InterestRate', 'BaseInterestRate', 'LoanAmount'

    - We now create a subset of the dataset with only these columns.

4) Train Test Split
    - We split the dataset in a **70:15:15 ratio** to create the Train, Validation and Test datasets.
    - We set the random state to 42 and apply stratification to ensure the proportion of approved and rejected records stay the same in all 3 datasets.

5) Encoding and Scaling
    - We now convert all the categorical columns to numeric using the **One Hot Encoder**.
    - We scale the data to ensure that all records in each column lie in withing similar and comparable range. 
    We used standard Scaler to scale the column values, since we had already applied log transformation to the data which stabalized the variance and reduced the severity of the outliers.
    We could have also used the **Robust Scaler** to scale the data values (since it is robust to outliers) if the log-transformed data still exhibited extreme values or heavy tails, which was not so in this case.

6) Balancing the data (approved and rejected loan application records)
    - The proportion of Approved v/s Rejected applications was nearly 30:70
    - Undersampling does mean loss of valuable data and we didn't use SMOTE to oversample the minority class as the classes were largely imbalanced.
    - Hence, we left the imbalace in the dataset as it is and instead focused on tuning the model parameters to better understand the non-linear relationships in the data.
   
7) Model Training
   - We have used 3 models - **Logistic Regression, Random Forest Classifier and XG Boost**.
   - We first evaluated the model on the validation set and on getting an optimal models (after perfoming hyperparameter tuning), we proceeded to predicting the test data.

8) Hyperparameter Tuning
    - We have used 2 methods to find the optimal combination of hyperparameters in each model:
        - **Grid Search CV for Logistic Regression**: As it there are few combinations of hyperparameter and this method tests evey possible combination of parameters. Hence, it won't take a long time to run.
        - **Random Search CV for Random Forest and XG Boost**: Since there are more combinations of parameters, testing out every combination would not only be time consuming but also computationally expensive. By using Random Search, we can test just a fraction of the combinations while capturing 95%+ of the potential performance boost.
     
9) Model Evaluation
   - We used the following evaluation metrics - **Precision, Recall, F1 Score and Confusion Matrix, ROC-AUC, PR-AUC**.
   - We have kept aside Accuracy for now, since this metric proves to be very misleading if the data contains imbalanced classes.
- Here is a summarized **Model Evaluation and Comparison Table** (Evaluation on the **Validation Dataset** (BEFORE TUNING)):
     | Model | Class Label | Precision | Recall | F1 Score |
     | ------------- | ------------- | ------------- | ------------- | ------------- |
     | Logistic Regression (base) | 0    |   0.94  |    0.95  |    0.95 |
     |   | 1    |   0.84   |   0.80    |  0.82 |
     | Random Forest (base) | 0    |   0.94  |    0.96  |    0.95 |
     | | 1   |    0.88   |   0.81   |   0.84 |
     | XG Boost (base) | 0   |    0.97  |    0.97   |   0.97 |
     |  |  1   |    0.91   |   0.91   |   0.91 |

    We can see that XGBoost identifies approximately 10% more churners than Logistic Regression while also maintaining higher precision.


- Here is a summarized **Model Evaluation and Comparison Table** (Evaluation on the **Validation Dataset** (AFTER TUNING)):
     | Model | Class Label | Precision | Recall | F1 Score |
     | ------------- | ------------- | ------------- | ------------- | ------------- |
     | Logistic Regression | 0    |   0.98  |    0.94  |    0.96 |
     |   | 1    |   0.82   |   0.95    |  0.88 |
     | Random Forest | 0    |   0.94  |    0.96  |    0.95 |
     | | 1   |    0.88   |   0.81   |   0.84 |
     | XG Boost | 0   |    0.97  |    0.97   |   0.97 |
     |  |  1   |    0.92   |   0.92   |   0.92 |

    We can see the improvement in the model perfromance on Logistic Regression and XG Boost models upon performing hyperparameter tuning. However, the performance of Random Forest seems to remain unchanged.  
    A possible reason to this may be that the RandomizedSearchCV only sampled a fraction of the grid (i.e., 15 out of 162 combinations), and hence, it may have missed to test the perfect combination of parameters.

- Here is a summarized **Model Evaluation and Comparison Table** (Evaluation on the **Test** Dataset):
     | Model | Class Label | Precision | Recall | F1 Score |
     | ------------- | ------------- | ------------- | ------------- | ------------- |
     | Logistic Regression | 0    |   0.98  |    0.94  |    0.96 |
     |   | 1    |   0.84   |   0.94    |  0.89 |
     | Random Forest | 0    |   0.93  |    0.97  |    0.95 |
     | | 1   |    0.89   |   0.78   |   0.83 |
     | XG Boost | 0   |    0.97  |    0.98   |   0.97 |
     |  |  1   |    0.93   |   0.90   |   0.92 |


- ROC-AUC and PR-AUC Scores of the Models:
     | Model | RUC-AUC | PR-AUC |
     | ------------- | ------------- | ------------- |
     | Logistic Regression | 0.9859 | 0.9644 |
     | Random Forest | 0.9758 | 0.9327 |
     | XG Boost | 0.9927 | 0.9787 |

<br><br>


<p align="center">
    <img src="./figures/image.png" alt="alt text" width="500">
    <img src="./figures/image-1.png" alt="alt text" width="500">
</p>

### Inference
   - We can rank the overall performance of the models in the following order:      
   **XG Boost > Logistic Regression > Random Forest Classifier**
   - Hence, **XG Boost** is our best performing model across all the evaluation metrics.
