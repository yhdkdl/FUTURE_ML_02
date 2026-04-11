import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.data.priority_labeler import get_priority_order
from src.utils.config import CATEGORY_TO_PRIORITY, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_classification_report_dict(
    y_true: pd.Series,
    y_pred,
    label_name: str,
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return report


def generate_markdown_report(
    category_metrics: dict,
    priority_metrics: dict,
    category_report: dict,
    priority_report: dict,
    category_model_name: str,
    priority_model_name: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Support Ticket Classifier - Evaluation Report
Generated: {now}

---

## Project Overview

An ML system that automatically classifies IT support tickets by:
- **Category** - what type of issue is it (7 classes)
- **Priority** - how urgent is it (4 levels)

Built using TF-IDF vectorization and Linear Support Vector Classification
trained on {1571} real IT support tickets from a Brazilian IT company.

---

## Dataset

| Property | Value |
|---|---|
| Source | Zenodo - Classification of IT Support Tickets |
| Total tickets | 2,229 |
| Training set | 1,571 |
| Test set | 657 |
| Languages | English, German, Portuguese, Spanish |
| Categories | 7 |
| Priority levels | 4 (rule-based assignment) |

---

## Category Classification Results

**Best Model: {category_model_name}**

| Metric | Score |
|---|---|
| Accuracy | {category_metrics['accuracy']} |
| Precision | {category_metrics['precision']} |
| Recall | {category_metrics['recall']} |
| F1 Score | {category_metrics['f1_score']} |

### Per-Class Performance

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
"""

    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    for label, scores in category_report.items():
        if label in skip_keys:
            continue
        if isinstance(scores, dict):
            md += (
                f"| {label} "
                f"| {scores['precision']:.3f} "
                f"| {scores['recall']:.3f} "
                f"| {scores['f1-score']:.3f} "
                f"| {int(scores['support'])} |\n"
            )

    md += f"""
---

## Priority Prediction Results

**Best Model: {priority_model_name}**

| Metric | Score |
|---|---|
| Accuracy | {priority_metrics['accuracy']} |
| Precision | {priority_metrics['precision']} |
| Recall | {priority_metrics['recall']} |
| F1 Score | {priority_metrics['f1_score']} |

### Per-Class Performance

| Priority | Precision | Recall | F1 | Support |
|---|---|---|---|---|
"""

    for label, scores in priority_report.items():
        if label in skip_keys:
            continue
        if isinstance(scores, dict):
            md += (
                f"| {label} "
                f"| {scores['precision']:.3f} "
                f"| {scores['recall']:.3f} "
                f"| {scores['f1-score']:.3f} "
                f"| {int(scores['support'])} |\n"
            )

    md += """
---

## Priority Assignment Rules

Priority is assigned based on ticket category using business rules:

| Category | Priority | Rationale |
|---|---|---|
"""

    rationale = {
        "EOL": "Planned decommission - not time critical",
        "Computer-Services": "Standard hardware/peripheral requests",
        "O365": "Software issues - affects productivity",
        "Software": "Software bugs - affects productivity",
        "Active Directory": "Blocks user access - needs prompt resolution",
        "Support general": "Escalated issues - needs prompt resolution",
        "Fileservice": "Blocks team file access - business critical",
    }

    for category, priority in CATEGORY_TO_PRIORITY.items():
        reason = rationale.get(category, "")
        md += f"| {category} | {priority} | {reason} |\n"

    md += """
---

## Key Insights

1. **LinearSVC outperforms** both Logistic Regression and Random Forest
   on this dataset - consistent with text classification literature

2. **Most common category confusions** are semantically reasonable:
   - Support general ↔ O365 (email tickets could belong to either)
   - Active Directory ↔ Support general (AD tickets often logged as general)
   - Software ↔ O365 (O365 is itself software)

3. **Multilingual tickets** (German, Portuguese, Spanish) are handled
   naturally by TF-IDF without any translation - foreign words become
   strong category signals

4. **Class imbalance** (EOL: 45 tickets vs Fileservice: 546) was handled
   with class_weight='balanced' - minority classes are still learned

5. **Priority model outperforms category model** (F1: 0.868 vs 0.787)
   because 4 classes are simpler to separate than 7

---

## Business Impact

| Before ML | After ML |
|---|---|
| Manual ticket reading | Automatic classification |
| Inconsistent prioritization | Rule-based consistent priority |
| Support team sorts tickets | Support team solves tickets |
| Response time: hours | Routing time: milliseconds |

---

## Figures

- outputs/figures/category_model_comparison.png
- outputs/figures/priority_model_comparison.png
- outputs/figures/confusion_matrix_svc.png
- outputs/figures/priority_confusion_matrix_svc.png
- outputs/figures/top_features_per_category.png
- outputs/figures/text_length_before_after.png
"""

    return md


def save_markdown_report(content: str, filename: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Markdown report saved to {filepath}")
    return filepath


def save_combined_metrics(
    category_metrics: dict,
    priority_metrics: dict,
    filename: str,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "category_classification": category_metrics,
        "priority_prediction": priority_metrics,
    }
    filepath = REPORTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined metrics saved to {filepath}")
    return filepath
