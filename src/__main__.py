from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import run


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Convert prompts into structured function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON function definition file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON prompt test file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where JSON results will be written.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name passed to Small_LLM_Model.",
    )
    return parser


def main() -> int:

    args = build_parser().parse_args()
    try:
        run(
            functions_path=Path(args.functions_definition),
            prompts_path=Path(args.input),
            output_path=Path(args.output),
            model_name=str(args.model),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
