"""
This file defines the structure of the project data.
It uses Pydantic models to validate the data.
It defines things like functions, prompts, and function calls.
It makes sure the data has the correct type and format.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonType = Literal["number", "integer", "string", "boolean", "array", "object"]


class ParameterDefinition(BaseModel):

    model_config = ConfigDict(extra="allow")

    type: JsonType
    enum: list[object] | None = None


class FunctionDefinition(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate a non-empty function name."""
        if not value.strip():
            raise ValueError("function name must not be empty")
        return value


class PromptCase(BaseModel):

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]
