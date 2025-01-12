import pytest
from typing import AsyncIterator, List
from llmrouter.base import Message, ChatResponse, CompletionResponse

def test_message_creation():
    """Test Message dataclass creation."""
    message = Message(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"

def test_chat_response_creation():
    """Test ChatResponse dataclass creation."""
    message = Message(role="assistant", content="Hello there!")
    response = ChatResponse(
        message=message,
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert response.message.content == "Hello there!"
    assert response.model == "test-model"
    assert response.usage["total_tokens"] == 15

def test_completion_response_creation():
    """Test CompletionResponse dataclass creation."""
    response = CompletionResponse(
        text="Test completion",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert response.text == "Test completion"
    assert response.model == "test-model"
    assert response.usage["total_tokens"] == 15 