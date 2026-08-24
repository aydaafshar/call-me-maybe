"""JSON file loading and writing helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.schemas import FunctionCall, FunctionDefinition, PromptCase


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file."""
    raw = _load_array(path)
    functions: list[FunctionDefinition] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                functions.append(FunctionDefinition.model_validate(item))
            except ValidationError:
                continue
    return functions


def load_prompts(path: Path) -> list[PromptCase]:
    """Load prompt cases from a JSON file."""
    raw = _load_array(path)
    prompts: list[PromptCase] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                prompts.append(PromptCase.model_validate(item))
            except ValidationError:
                continue
    return prompts


def write_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write generated function calls as valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "prompt": call.prompt,
            "name": call.name,
            "parameters": call.parameters,
        }
        for call in calls
    ]
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def _load_array(path: Path) -> list[object]:
    """Load a JSON array, returning an empty list for malformed input."""
    if not path.exists():
        print(f"Error: file not found: {path}")
        return []

    try:
        with path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in file: {path}")
        return []
    except OSError as error:
        print(f"Error: could not read file {path}: {error}")
        return []

    if not isinstance(data, list):
        print(f"Error: expected a JSON array in file: {path}")
        return []

    return data
