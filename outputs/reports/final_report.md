# Support Ticket Classifier - Evaluation Report
Generated: 2026-04-11 11:44

---

## Project Overview

An ML system that automatically classifies IT support tickets by:
- **Category** - what type of issue is it (7 classes)
- **Priority** - how urgent is it (4 levels)

Built using TF-IDF vectorization and Linear Support Vector Classification
trained on 1571 real IT support tickets from a Brazilian IT company.

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

**Best Model: LinearSVC**

| Metric | Score |
|---|---|
| Accuracy | 0.7839 |
| Precision | 0.7963 |
| Recall | 0.7839 |
| F1 Score | 0.7874 |

### Per-Class Performance

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Active Directory | 0.472 | 0.510 | 0.490 | 49 |
| Computer-Services | 0.694 | 0.610 | 0.649 | 41 |
| EOL | 1.000 | 1.000 | 1.000 | 58 |
| Fileservice | 0.962 | 0.920 | 0.941 | 138 |
| O365 | 0.595 | 0.783 | 0.676 | 92 |
| Software | 0.691 | 0.655 | 0.673 | 58 |
| Support general | 0.842 | 0.769 | 0.804 | 221 |

---

## Priority Prediction Results

**Best Model: LinearSVC**

| Metric | Score |
|---|---|
| Accuracy | 0.8676 |
| Precision | 0.8678 |
| Recall | 0.8676 |
| F1 Score | 0.8676 |

### Per-Class Performance

| Priority | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Critical | 0.956 | 0.935 | 0.945 | 138 |
| High | 0.845 | 0.867 | 0.856 | 270 |
| Low | 1.000 | 1.000 | 1.000 | 58 |
| Medium | 0.797 | 0.780 | 0.788 | 191 |

---

## Priority Assignment Rules

Priority is assigned based on ticket category using business rules:

| Category | Priority | Rationale |
|---|---|---|
| EOL | Low | Planned decommission - not time critical |
| Computer-Services | Medium | Standard hardware/peripheral requests |
| O365 | Medium | Software issues - affects productivity |
| Software | Medium | Software bugs - affects productivity |
| Active Directory | High | Blocks user access - needs prompt resolution |
| Support general | High | Escalated issues - needs prompt resolution |
| Fileservice | Critical | Blocks team file access - business critical |

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
