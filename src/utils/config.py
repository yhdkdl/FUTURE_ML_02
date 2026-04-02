from pathlib import Path

# Root of the project — resolves to the folder containing /src
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data paths
RAW_DATA_DIR    = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Output paths
MODELS_DIR   = ROOT_DIR / "outputs" / "models"
REPORTS_DIR  = ROOT_DIR / "outputs" / "reports"
FIGURES_DIR  = ROOT_DIR / "outputs" / "figures"

# Target columns — single source of truth
CATEGORY_TARGET = "Ticket Type"
PRIORITY_TARGET  = "Ticket Priority"
TEXT_COLUMN      = "Ticket Description"

# Exact class labels from EDA — Sprint 1 confirmed
CATEGORY_LABELS = [
    "Refund request",
    "Technical issue",
    "Cancellation request",
    "Product inquiry",
    "Billing inquiry"
]

PRIORITY_LABELS = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

# Priority is ordinal — this ordering matters for Sprint 5
# Low=0, Medium=1, High=2, Critical=3
PRIORITY_ORDER = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}