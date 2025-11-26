from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .agent import run_interaction
from .config import load_settings
from .observability import configure_langsmith


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run DevMate agent")
    parser.add_argument("query", help="User request to handle")
    args = parser.parse_args()

    settings = load_settings()
    configure_langsmith(settings)

    output = run_interaction(settings, args.query)
    print(output)


if __name__ == "__main__":
    main()
