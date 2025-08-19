# Financial-Risk-for-Loan-Approval

### Business Problem: Optimizing Loan Portfolio Profitability While Minimizing Risk Exposure
**Background:**    
A financial institution wants to grow its loan portfolio by approving more applications without significantly increasing its default risk. Historically, overly cautious approval strategies have led to lost revenue opportunities, while lenient strategies have increased default rates and write-offs.

**Objective:**    
Use predictive modeling to identify applicants who are both likely to repay their loans and profitable to approve, balancing approval rates with financial risk and expected returns.

#### Key Question to Address:
<!-- ##### Risk Assessment (Regression Task)

1) What is the predicted RiskScore for a new applicant, indicating the likelihood of default?

2) How does this score correlate with key variables like credit history, debt-to-income ratio, or loan purpose? -->

**Loan Approval (Classification Task)** - Should this applicant be approved or denied for a loan based on their financial profile?

<!-- ##### Strategic Decisioning

1) How can we segment applicants (e.g., low-risk/high-value, high-risk/low-value) to refine lending strategies?

2) What threshold on the RiskScore corresponds to optimal approval criteria (balancing profit vs. risk)?

3) Can we dynamically adjust interest rates based on predicted risk and expected loss? -->


### Dataset
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


### Implementation
1) Understanding the Dataset (performing inintal analysis)
    - Overall shape of the dataset (20000,36)
    - Presence of any null values (no null values)
    - Checking the datatypes of the columns (float, int and object)
    - Identifying calculated columns so as to retain those and drop the component columns
        (for instance, there is a annualIncome column which is MonthlyIncome * 12. So, Instead of retaining both the monthly and annual income columns, we can just retain the annual income column.)

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
        -  (having positive correlation) 'NetWorth', 'CreditScore', 'Age', 'Experience', 'LengthOfCreditHistory'
        -  (having negative correlation) 'DebtToIncomeRatio', 'Monthly_LoanToIncomeRatio', 'RiskScore'

    - We now create a subset of the dataset with only these columns.

4) Train Test Split
    - We split the dataset in a **60:20:20 ratio** to create the Train, Validation and Test datasets.
    - We set the random state to 42 and apply stratification to ensure the proportion of approved and rejected records stay the same in all 3 datasets.

5) Encoding and Scaling
    - We now convert all the categorical columns to numeric using the **Ordinal Encoder**.
    - To scale the data to ensure that all records in each column lie in the same range. We use the **Robust Scaler** to scale the data values since it is robust to outliers.  

6) Balancing the data (approved and rejected loan application records)
    - We have performed Random Under Sampling to balance the number of records from the 2 labels
    - The proportion of Approved v/s Rejected applications was nearly 25:75
    - Undersampling does mean loss of valuable data but we did use SMOTE to oversample the minority class but it didn't result in a good model performance.
    - Hence, we resorted to undersampling as the models performed better during the evaluation process.
   
7) Model Training
   - We have used 3 models - **Logistic Regression, Random Forest Classifier and XG Boost**.
   - We first evaluated the model on the Validation set and on getting a good result, we proceeded to predicting the Test data.
     
9) Model Evaluation
   - We used the following evaluation metrics - **Precision, Recall, F1 Score and Confusion Matrix**.
   - Here is a summarized **Model Evaluation and Comparison Table** (Evaluation on the Test Dataset):
     | Model | Class Label | Precision | Recall | F1 Score |
     | ------------- | ------------- | ------------- | ------------- | ------------- |
     | Logistic Regression | 0    |   0.95  |    0.86  |    0.90 |
     |   | 1    |   0.66   |   0.86    |  0.75 |
     | Random Forest | 0    |   0.99  |    0.98  |    0.99 |
     | | 1   |    0.93   |   0.98   |   0.96 |
     | XG Boost | 0   |    1.00  |    0.98   |   0.99 |
     |  |  1   |    0.94   |   0.99   |   0.97 |


   - We can rank the overall performance of the models in the following order: **XG Boost > Random Forest Classifier > Logistic Regression**
   - Hence, **XG Boost** is our best performing model.
