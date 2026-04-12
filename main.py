import argparse
import sys

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Support Ticket Classifier - "
            "Automatically classify IT support tickets "
            "by category and priority."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticket "Cannot access shared drive"
  python main.py --file tickets.csv
  python main.py --file tickets.csv --column description
  python main.py --demo
  python main.py --info
        """,
    )

    parser.add_argument(
        "--ticket",
        "-t",
        type=str,
        help="Classify a single ticket (wrap in quotes)",
        metavar="TEXT",
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to a CSV file of tickets to classify",
        metavar="PATH",
    )

    parser.add_argument(
        "--column",
        "-c",
        type=str,
        default="text",
        help="Column name for ticket text in CSV (default: text)",
        metavar="COLUMN",
    )

    parser.add_argument(
        "--demo",
        "-d",
        action="store_true",
        help="Run a live demo with sample tickets",
    )

    parser.add_argument(
        "--info",
        "-i",
        action="store_true",
        help="Show system info and model performance",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Show help if no arguments provided.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    from src.cli.commands import cmd_demo, cmd_info, cmd_predict_file, cmd_predict_ticket

    if args.info:
        cmd_info()
    elif args.demo:
        cmd_demo()
    elif args.ticket:
        cmd_predict_ticket(args.ticket)
    elif args.file:
        cmd_predict_file(args.file, args.column)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()