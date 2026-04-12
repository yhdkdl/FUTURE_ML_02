import sys
import json
from collections import Counter
from pathlib import Path

from src.utils.config import (
    REPORTS_DIR,
    MODELS_DIR,
    CATEGORY_LABELS,
    PRIORITY_LABELS,
    CATEGORY_TO_PRIORITY,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cmd_predict_ticket(ticket_text: str) -> None:
    from src.pipeline.predictor import predict_ticket

    if not ticket_text.strip():
        print("ERROR: Ticket text cannot be empty.")
        sys.exit(1)

    print("\nClassifying ticket...")
    result = predict_ticket(ticket_text)
    print(result)


def cmd_predict_file(filepath: str, text_column: str = "text") -> None:
    from src.pipeline.predictor import TicketPredictor
    import pandas as pd

    path = Path(filepath)

    if not path.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    if path.suffix.lower() != ".csv":
        print(f"ERROR: Only CSV files are supported. Got: {path.suffix}")
        sys.exit(1)

    print(f"\nLoading tickets from {filepath}...")
    df = pd.read_csv(path)

    if text_column not in df.columns:
        print(f"ERROR: Column '{text_column}' not found in CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"Found {len(df)} tickets. Classifying...")

    predictor = TicketPredictor().load()
    results_df = predictor.predict_dataframe(df, text_column)

    results_df_sorted = results_df.sort_values("priority_level", ascending=False)

    print("\n=== CLASSIFIED TICKET QUEUE (sorted by priority) ===\n")
    print(f"{'Ticket':<50} {'Category':<22} {'Priority'}")
    print("-" * 85)

    for _, row in results_df_sorted.iterrows():
        text = str(row[text_column])[:48]
        category = row["predicted_category"]
        priority = row["predicted_priority"]
        print(f"{text:<50} {category:<22} {priority}")

    output_path = path.parent / f"{path.stem}_classified.csv"
    results_df_sorted.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Total tickets classified: {len(results_df)}")


def cmd_info() -> None:
    print(
        """
╔══════════════════════════════════════════════════════╗
║         Support Ticket Classifier — System Info      ║
╚══════════════════════════════════════════════════════╝

📊 MODEL PERFORMANCE
  Category Classification (7 classes)
        • Model:    LinearSVC
        • Accuracy: 78.4%
        • F1 Score: 0.787

  Priority Prediction (4 levels)
        • Model:    LinearSVC
        • Accuracy: 86.8%
        • F1 Score: 0.868

📁 DATASET
    • Source:   Zenodo — IT Support Tickets
    • Train:    1,571 tickets
    • Test:     657 tickets
    • Languages: English, German, Portuguese, Spanish

🏷️  CATEGORIES
"""
    )

    for label in CATEGORY_LABELS:
        priority = CATEGORY_TO_PRIORITY.get(label, "Medium")
        print(f"  • {label:<25} → {priority}")

    print(
        """

🚨 PRIORITY LEVELS
    • Critical  (Level 3) — Business critical, immediate action
    • High      (Level 2) — Affects user access, prompt resolution
    • Medium    (Level 1) — Productivity impact, same day
    • Low       (Level 0) — Planned work, scheduled resolution

📂 MODEL FILES
"""
    )

    model_files = [
        "tfidf_vectorizer.pkl",
        "category_classifier.pkl",
        "priority_classifier.pkl",
    ]

    for filename in model_files:
        path = MODELS_DIR / filename
        status = "✅" if path.exists() else "❌ MISSING"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else ""
        print(f"  {status}  {filename} {size}")

    print()


def cmd_demo() -> None:
    from src.pipeline.predictor import TicketPredictor

    demo_tickets = [
        "Cannot access shared network drive, permission denied error",
        "Need to create Active Directory account for new employee",
        "Outlook not syncing emails with O365 mailbox",
        "Application crashing with null pointer on Windows 11",
        "Office printer showing offline, cannot print documents",
        "Decommission old server, remove from Nexthink inventory",
        "Employee resigned, disable all system access immediately",
        "VPN connection dropping intermittently from home office",
    ]

    print(
        """
╔══════════════════════════════════════════════════════╗
║        Support Ticket Classifier — Live Demo         ║
╚══════════════════════════════════════════════════════╝
"""
    )

    print("Loading models...")
    predictor = TicketPredictor().load()

    print(f"\nClassifying {len(demo_tickets)} sample tickets...\n")
    results = predictor.predict_batch(demo_tickets)

    results_sorted = sorted(results, key=lambda r: r.priority_level, reverse=True)

    print(f"{'#':<4} {'Priority':<12} {'Category':<22} {'Ticket'}")
    print("─" * 90)

    for i, result in enumerate(results_sorted, 1):
        priority_icons = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢",
        }
        icon = priority_icons.get(result.priority, "⚪")
        print(
            f"{i:<4} "
            f"{icon} {result.priority:<10} "
            f"{result.category:<22} "
            f"{result.raw_text[:45]}"
        )

    print("\n" + "─" * 90)

    priority_counts = Counter(r.priority for r in results)

    print("\n📊 SUMMARY")
    for priority in ["Critical", "High", "Medium", "Low"]:
        count = priority_counts.get(priority, 0)
        bar = "█" * count
        print(f"  {priority:<10} {bar} ({count})")

    print("\n✅ Demo complete.\n")