from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["Message", "ChatRequest"]


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
