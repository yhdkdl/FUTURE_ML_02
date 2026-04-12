import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import FIGURES_DIR, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_test,
    y_test: pd.Series,
    model_name: str,
) -> tuple[dict, np.ndarray]:
    logger.info(f"Evaluating {model_name}...")

    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0),
            4,
        ),
        "recall": round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0),
            4,
        ),
        "f1_score": round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0),
            4,
        ),
    }

    logger.info(f"  Accuracy:  {metrics['accuracy']}")
    logger.info(f"  Precision: {metrics['precision']}")
    logger.info(f"  Recall:    {metrics['recall']}")
    logger.info(f"  F1 Score:  {metrics['f1_score']}")

    return metrics, y_pred


def print_classification_report(
    y_test: pd.Series,
    y_pred,
    model_name: str,
) -> None:
    print(f"\n=== CLASSIFICATION REPORT — {model_name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred,
    model_name: str,
    labels: list,
    filename: str,
) -> None:
    y_true_labels = set(pd.Series(y_test).unique().tolist())
    y_pred_labels = set(pd.Series(y_pred).tolist())
    label_order = [label for label in labels if label in y_true_labels or label in y_pred_labels]

    if not label_order:
        raise ValueError(
            "No labels overlap between the provided labels list and the actual data. "
            f"Provided labels: {labels}. "
            f"y_true labels: {sorted(y_true_labels)}. "
            f"y_pred labels: {sorted(y_pred_labels)}."
        )

    cm = confusion_matrix(y_test, y_pred, labels=label_order)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} — Raw Counts", fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order,
        ax=axes[1],
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"{model_name} — Normalized", fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Confusion matrix saved to {filepath}")


def save_metrics_report(all_metrics: list, filename: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics report saved to {filepath}")


def plot_model_comparison(all_metrics: list, filename: str) -> None:
    df = pd.DataFrame(all_metrics)
    metrics_cols = ["accuracy", "precision", "recall", "f1_score"]

    df_melted = df.melt(
        id_vars="model",
        value_vars=metrics_cols,
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_melted,
        x="metric",
        y="score",
        hue="model",
        ax=ax,
        palette="Set2",
    )

    ax.set_title("Model Comparison — Category Classification", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Comparison chart saved to {filepath}")