from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    FunctionCall,
    FunctionDefinition,
    ParameterDefinition,
    PromptCase,
)


def test_parameter_definition_minimal() -> None:
    param = ParameterDefinition(type="string")
    assert param.type == "string"
    assert param.enum is None


def test_parameter_definition_with_enum() -> None:
    param = ParameterDefinition(type="string", enum=["a", "b"])
    assert param.enum == ["a", "b"]


def test_parameter_definition_allows_extra_fields() -> None:
    param = ParameterDefinition(  # type: ignore[call-arg]
        type="number", description="extra info"
    )
    assert param.description == "extra info"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "json_type",
    ["number", "integer", "string", "boolean", "array", "object"],
)
def test_parameter_definition_accepts_every_json_type(json_type: str) -> None:
    param = ParameterDefinition(type=json_type)  # type: ignore[arg-type]
    assert param.type == json_type


def test_parameter_definition_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ParameterDefinition(type="float")  # type: ignore[arg-type]


def test_function_definition_defaults() -> None:
    func = FunctionDefinition(name="get_weather")
    assert func.description == ""
    assert func.parameters == {}
    assert func.returns == {}


def test_function_definition_with_parameters() -> None:
    func = FunctionDefinition(
        name="get_weather",
        parameters={"city": ParameterDefinition(type="string")},
    )
    assert func.parameters["city"].type == "string"


def test_function_definition_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition(  # type: ignore[call-arg]
            name="get_weather", unexpected_field=True
        )


def test_function_definition_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition(name="")


def test_function_definition_rejects_whitespace_only_name() -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition(name="   ")


def test_prompt_case_holds_prompt() -> None:
    case = PromptCase(prompt="What's the weather?")
    assert case.prompt == "What's the weather?"


def test_prompt_case_ignores_extra_fields() -> None:
    case = PromptCase(prompt="hello", extra_field="ignored")  # type: ignore[call-arg]
    assert not hasattr(case, "extra_field")


def test_prompt_case_requires_prompt() -> None:
    with pytest.raises(ValidationError):
        PromptCase()  # type: ignore[call-arg]


def test_function_call_valid() -> None:
    call = FunctionCall(
        prompt="What's the weather in Paris?",
        name="get_weather",
        parameters={"city": "Paris"},
    )
    assert call.name == "get_weather"
    assert call.parameters == {"city": "Paris"}


def test_function_call_allows_empty_parameters() -> None:
    call = FunctionCall(prompt="hi", name="noop", parameters={})
    assert call.parameters == {}


def test_function_call_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FunctionCall(
            prompt="hi",
            name="noop",
            parameters={},
            confidence=0.9,  # type: ignore[call-arg]
        )


def test_function_call_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        FunctionCall(name="noop", parameters={})  # type: ignore[call-arg]
