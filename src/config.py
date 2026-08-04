"""Project-wide paths and modelling constants."""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

RANDOM_STATE = 42
TARGET_CLASSIFICATION = "LoanApproved"

root_path_env = os.getenv("ROOT_PATH")
# Setting a local-path fallback incase Path(os.getenv("ROOT_PATH")) crashes.
PROJECT_ROOT = Path(root_path_env)  if root_path_env else Path.cwd()
DATA_DIR_PATH = PROJECT_ROOT / "data files"
DATA_FILE = PROJECT_ROOT / "data files" / "Loan.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "trained models" / "pipeline"

# Exclude variables unavailable at the time of an approval decision, or likely
# to encode the decision itself. Confirm this with the business before changing.
LEAKAGE_OR_UNUSED_COLUMNS = ["RiskScore"]