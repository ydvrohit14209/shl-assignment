"""
API contract for the SHL assessment recommendation chat service.

POST /chat is stateless: the client sends the *entire* conversation so far
on every call, and the server returns the assistant's next turn. The server
holds no session state between requests (no DB, no in-memory dict keyed by
session id) — this matches the assignment's statelessness requirement and
means the service is trivially horizontally scalable / restart-safe.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[Message]) -> list[Message]:
        if v[-1].role != Role.user:
            raise ValueError("the last message in the conversation must be from the user")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: list[str]


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
