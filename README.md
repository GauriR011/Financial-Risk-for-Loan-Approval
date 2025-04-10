# Financial-Risk-for-Loan-Approval

### Business Problem: Optimizing Loan Portfolio Profitability While Minimizing Risk Exposure
Background:
A financial institution wants to grow its loan portfolio by approving more applications without significantly increasing its default risk. Historically, overly cautious approval strategies have led to lost revenue opportunities, while lenient strategies have increased default rates and write-offs.

#### Objective:
Use predictive modeling to identify applicants who are both likely to repay their loans and profitable to approve, balancing approval rates with financial risk and expected returns.

#### Key Questions to Address:
##### Risk Assessment (Regression Task)

1) What is the predicted RiskScore for a new applicant, indicating the likelihood of default?

2) How does this score correlate with key variables like credit history, debt-to-income ratio, or loan purpose?

##### Loan Approval (Classification Task)

1) Should this applicant be approved or denied for a loan based on their financial profile?

2) Can we automate or assist underwriters in making consistent, fair, and data-driven decisions?

##### Strategic Decisioning

1) How can we segment applicants (e.g., low-risk/high-value, high-risk/low-value) to refine lending strategies?

2) What threshold on the RiskScore corresponds to optimal approval criteria (balancing profit vs. risk)?

3) Can we dynamically adjust interest rates based on predicted risk and expected loss?


### Dataset
The dataset includes the following columns:

ApplicationDate: Loan application date
Age: Applicant's age
AnnualIncome: Yearly income
CreditScore: Creditworthiness score
EmploymentStatus: Job situation
EducationLevel: Highest education attained
Experience: Work experience
LoanAmount: Requested loan size
LoanDuration: Loan repayment period
MaritalStatus: Applicant's marital state
NumberOfDependents: Number of dependents
HomeOwnershipStatus: Homeownership type
MonthlyDebtPayments: Monthly debt obligations
CreditCardUtilizationRate: Credit card usage percentage
NumberOfOpenCreditLines: Active credit lines
NumberOfCreditInquiries: Credit checks count
DebtToIncomeRatio: Debt to income proportion
BankruptcyHistory: Bankruptcy records
LoanPurpose: Reason for loan
PreviousLoanDefaults: Prior loan defaults
PaymentHistory: Past payment behavior
LengthOfCreditHistory: Credit history duration
SavingsAccountBalance: Savings account amount
CheckingAccountBalance: Checking account funds
TotalAssets: Total owned assets
TotalLiabilities: Total owed debts
MonthlyIncome: Income per month
UtilityBillsPaymentHistory: Utility payment record
JobTenure: Job duration
NetWorth: Total financial worth
BaseInterestRate: Starting interest rate
InterestRate: Applied interest rate
MonthlyLoanPayment: Monthly loan payment
TotalDebtToIncomeRatio: Total debt against income
LoanApproved: Loan approval status
RiskScore: Risk assessment score