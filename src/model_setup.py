"""Model construction and instruction-prompt preparation."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import cast

from llm_sdk import Small_LLM_Model

from src.function_call_builder import generate_function_call
from src.schemas import FunctionCall, FunctionDefinition
from src.vocabulary import load_vocabulary


def create_model(model_name: str) -> tuple[object, dict[int, str]]:
    """Create the Small_LLM_Model wrapper and load its vocabulary."""
    model = Small_LLM_Model(model_name=model_name)
    vocabulary = load_vocabulary(_vocabulary_path(model))
    return model, vocabulary


def choose_function_call(
    model: object,
    vocabulary: dict[int, str],
    prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCall:
    """Generate a schema-valid function call for one prompt."""
    context_prompt = _selection_prompt(prompt, functions)
    return generate_function_call(model, vocabulary, context_prompt, prompt, functions)


def _selection_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the natural-language context the model reasons over."""
    definitions = [
        {
            "name": function.name,
            "description": function.description,
            "parameters": {
                name: definition.model_dump(exclude_none=True)
                for name, definition in function.parameters.items()
            },
        }
        for function in functions
    ]
    return (
        "Select exactly one JSON function call for the user prompt. "
        "Do not answer the prompt.\nFunctions:\n"
        f"{json.dumps(definitions, ensure_ascii=False)}\nPrompt:\n{prompt}\nJSON:"
    )


def _vocabulary_path(model: object) -> Path:
    """Return a public SDK vocabulary path, accepting SDK version differences."""
    for method_name in (
        "get_path_to_vocab_file",
        "get_path_to_vocabulary_json",
        "get_path_to_tokenizer_file",
    ):
        getter = getattr(model, method_name, None)
        if callable(getter):
            typed_getter = cast(Callable[[], str], getter)
            return Path(typed_getter())
    raise TypeError("model does not provide a public vocabulary path method")
