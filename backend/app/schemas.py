from typing import Literal

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9-]{8,64}$"


class Artifact(BaseModel):
    type: Literal["letter", "guide"]
    text: str
    title: str = ""


class AgentResult(BaseModel):
    reply: str
    tools_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    citation_note: str = Field(default="")
    provider: str = Field(default="gemini")


class ChatIn(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message: str = Field(min_length=1, max_length=1000)
    lang: Literal["en", "hi"] = "en"


class VoiceOut(AgentResult):
    transcript: str


class TtsIn(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    lang: str = "hi"
