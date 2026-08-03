"""Project-wide paths and modelling constants."""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

RANDOM_STATE = 42
TARGET_CLASSIFICATION = "LoanApproved"

PROJECT_ROOT = Path(os.getenv("ROOT_PATH"))
DATA_DIR_PATH = PROJECT_ROOT / "data files"
DATA_FILE = PROJECT_ROOT / "data files" / "Loan.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "trained models"

# Exclude variables unavailable at the time of an approval decision, or likely
# to encode the decision itself. Confirm this with the business before changing.
LEAKAGE_OR_UNUSED_COLUMNS = ["RiskScore"]