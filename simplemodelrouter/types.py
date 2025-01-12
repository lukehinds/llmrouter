from typing import Any, List, Optional
from pydantic import BaseModel


class Delta(BaseModel):
    """Delta represents a change in content for streaming responses."""
    content: Optional[str] = None
    role: Optional[str] = None


class Choice(BaseModel):
    """Choice represents a single completion choice from the model."""
    finish_reason: Optional[str] = None
    index: int = 0
    delta: Delta
    logprobs: Optional[Any] = None


class Response(BaseModel):
    """Response represents a model's response to a prompt."""
    id: str
    choices: List[Choice]
    created: int  # Unix timestamp
    model: str
    object: str = "chat.completion.chunk"
    stream: bool = False

    def json(self) -> str:
        """Convert the response to a JSON string."""
        return self.model_dump_json(exclude_none=True)

    @property
    def message(self) -> Optional[Choice]:
        """Get the first choice from the response."""
        return self.choices[0] if self.choices else None