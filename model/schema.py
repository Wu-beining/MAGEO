from typing import Any, Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[dict[str, Any]] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class ToolCall(BaseModel):
    id: str
    type: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, Any] | None = None

