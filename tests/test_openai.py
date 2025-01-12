import pytest
import httpx
import json
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

from simplemodelrouter import OpenAIProvider, Message, ChatResponse, CompletionResponse

@pytest.fixture
def mock_response():
    """Create a mock response for testing."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Test response"
            }
        }],
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }

@pytest.fixture
def mock_completion_response():
    """Create a mock completion response for testing."""
    return {
        "choices": [{
            "text": "Test completion"
        }],
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }

@pytest.mark.asyncio
async def test_chat(mock_response):
    """Test chat method."""
    provider = OpenAIProvider(api_key="test-key")
    
    with patch.object(httpx.AsyncClient, 'post') as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.json.return_value = mock_response
        
        messages = [Message(role="user", content="Hello")]
        response = await provider.chat(messages)
        
        assert isinstance(response, ChatResponse)
        assert response.message.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        
        await provider.close()

@pytest.mark.asyncio
async def test_complete(mock_completion_response):
    """Test complete method."""
    provider = OpenAIProvider(api_key="test-key")
    
    with patch.object(httpx.AsyncClient, 'post') as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.json.return_value = mock_completion_response
        
        response = await provider.complete("Hello")
        
        assert isinstance(response, CompletionResponse)
        assert response.text == "Test completion"
        assert response.model == "gpt-3.5-turbo"
        
        await provider.close()

@pytest.mark.asyncio
async def test_streaming_chat():
    """Test streaming chat responses."""
    provider = OpenAIProvider(api_key="test-key")
    
    stream_data = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"}}],"model":"gpt-3.5-turbo"}\n',
        'data: {"choices":[{"delta":{"content":" world"}}],"model":"gpt-3.5-turbo"}\n',
        'data: [DONE]\n'
    ]
    
    with patch.object(httpx.AsyncClient, 'stream') as mock_stream:
        mock_stream.return_value.__aenter__.return_value.aiter_lines = AsyncMock(
            return_value=aiter(stream_data)
        )
        
        messages = [Message(role="user", content="Hi")]
        response_stream = await provider.chat(messages, stream=True)
        
        responses = [r async for r in response_stream]
        assert len(responses) == 2
        assert responses[0].message.content == "Hello"
        assert responses[1].message.content == " world"
        
        await provider.close()

@pytest.mark.asyncio
async def test_streaming_complete():
    """Test streaming completion responses."""
    provider = OpenAIProvider(api_key="test-key")
    
    stream_data = [
        'data: {"choices":[{"text":"Hello"}],"model":"gpt-3.5-turbo"}\n',
        'data: {"choices":[{"text":" world"}],"model":"gpt-3.5-turbo"}\n',
        'data: [DONE]\n'
    ]
    
    with patch.object(httpx.AsyncClient, 'stream') as mock_stream:
        mock_stream.return_value.__aenter__.return_value.aiter_lines = AsyncMock(
            return_value=aiter(stream_data)
        )
        
        response_stream = await provider.complete("Hi", stream=True)
        
        responses = [r async for r in response_stream]
        assert len(responses) == 2
        assert responses[0].text == "Hello"
        assert responses[1].text == " world"
        
        await provider.close() 