from __future__ import annotations

import json

from src.schemas import (
    FunctionCall,
    FunctionDefinition,
    JsonType,
    ParameterDefinition,
)
from src.token_io import encode
from src.value_decoding import choose_literal, generate_number, generate_string


def generate_function_call(
    model: object,
    vocabulary: dict[int, str],
    context_prompt: str,
    prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCall:

    ids = encode(model, context_prompt)
    ids += encode(model, '{"prompt":' + json.dumps(prompt) + ',"name":"')
    name, name_ids = choose_literal(model, ids, [fn.name for fn in functions])
    ids += name_ids
    function_def = next(fn for fn in functions if fn.name == name)
    ids += encode(model, '","parameters":{')
    parameters: dict[str, object] = {}
    items = list(function_def.parameters.items())
    if not items:
        ids += encode(model, "}}")
    for index, (param_name, param_def) in enumerate(items):
        ids += encode(model, json.dumps(param_name) + ":")
        separator = "," if index < len(items) - 1 else "}}"
        value, value_ids = _generate_value(
            model,
            ids,
            vocabulary,
            param_def,
            separator,
        )
        parameters[param_name] = value
        ids += value_ids
    return FunctionCall(prompt=prompt, name=name, parameters=parameters)


def default_for_type(type_: JsonType) -> object:
    """Return a schema-valid default value for a parameter type."""
    defaults: dict[str, object] = {
        "number": 0,
        "integer": 0,
        "string": "",
        "boolean": False,
        "array": [],
        "object": {},
    }
    return defaults.get(type_)


def _generate_value(
    model: object,
    ids: list[int],
    vocabulary: dict[int, str],
    param_def: ParameterDefinition,
    separator: str,
) -> tuple[object, list[int]]:

    if param_def.enum:
        options = _json_literals(param_def.enum)
        literal, literal_ids = choose_literal(model, ids, options)
        return json.loads(literal), literal_ids + encode(model, separator)
    if param_def.type == "boolean":
        literal, literal_ids = choose_literal(model, ids, ["true", "false"])
        return literal == "true", literal_ids + encode(model, separator)
    if param_def.type in {"number", "integer"}:
        return generate_number(
            model,
            ids,
            vocabulary,
            separator,
            integer_only=param_def.type == "integer",
        )
    if param_def.type == "string":
        return generate_string(model, ids, vocabulary, separator)
    literal = "[]" if param_def.type == "array" else "{}"
    return default_for_type(param_def.type), encode(model, literal + separator)


def _json_literals(values: list[object]) -> list[str]:
    """converts enum values into valid JSON text."""
    return [json.dumps(value, ensure_ascii=False) for value in values]
