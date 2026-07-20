"""Application orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

from src.file_io import load_functions, load_prompts, write_results
from src.function_call_builder import default_for_type
from src.model_setup import choose_function_call, create_model
from src.schemas import FunctionCall, FunctionDefinition, PromptCase


def run(
    functions_path: Path,
    prompts_path: Path,
    output_path: Path,
    model_name: str,
) -> None:
    """Read inputs, generate calls, and write JSON output."""
    functions = load_functions(functions_path)
    prompts = load_prompts(prompts_path)
    results = generate_results(functions, prompts, model_name)
    write_results(output_path, results)


def generate_results(
    functions: list[FunctionDefinition],
    prompts: list[PromptCase],
    model_name: str,
) -> list[FunctionCall]:
    """Generate function calls for all prompt cases."""
    if not functions or not prompts:
        return []
    model, vocabulary = create_model(model_name)
    results: list[FunctionCall] = []
    for prompt_case in prompts:
        try:
            results.append(
                choose_function_call(model, vocabulary, prompt_case.prompt, functions)
            )
        except Exception as exc:
            print(f"warning: generation failed for a prompt: {exc}", file=sys.stderr)
            results.append(_fallback_call(prompt_case.prompt, functions[0]))
    return results


def _fallback_call(prompt: str, function: FunctionDefinition) -> FunctionCall:
    """Build a schema-valid placeholder call when generation fails unexpectedly."""
    parameters = {
        name: default_for_type(definition.type)
        for name, definition in function.parameters.items()
    }
    return FunctionCall(prompt=prompt, name=function.name, parameters=parameters)
