from pathlib import Path

# Root of the project
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data paths
RAW_DATA_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Output paths
MODELS_DIR  = ROOT_DIR / "outputs" / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

# Column names — Zenodo dataset
TEXT_COLUMN      = "text"
CATEGORY_TARGET  = "category_truth"
ID_COLUMN        = "id"

# Column used in processed dataset notebooks
PRIORITY_TARGET  = "Ticket Priority"

# Exact category labels confirmed from EDA
CATEGORY_LABELS = [
    "Fileservice",
    "Support general",
    "Software",
    "O365",
    "Active Directory",
    "Computer-Services",
    "EOL"
]

# Priority labels — rule-based assignment Sprint 5
# These map category → priority based on IT support logic
PRIORITY_LABELS = ["Low", "Medium", "High", "Critical"]

CATEGORY_TO_PRIORITY = {
    "EOL":               "Low",
    "Computer-Services": "Medium",
    "O365":              "Medium",
    "Software":          "Medium",
    "Active Directory":  "High",
    "Support general":   "High",
    "Fileservice":       "Critical",
}

PRIORITY_ORDER = {
    "Low":      0,
    "Medium":   1,
    "High":     2,
    "Critical": 3
}