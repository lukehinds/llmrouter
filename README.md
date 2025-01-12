# LLMRouter

A Python library for interfacing with various Large Language Model (LLM) inference endpoints, including OpenAI, Anthropic, and Ollama. The library provides a unified, async-first interface for interacting with different LLM providers.

## Features

- Support for multiple LLM providers:
  - OpenAI (GPT-3.5, GPT-4)
  - Anthropic (Claude)
  - Ollama (Local models)
- Async HTTP support using httpx
- Streaming responses for real-time text generation
- Unified interface across providers
- Type hints and comprehensive documentation
- Configurable API endpoints and models
- Error handling and retries
- Resource cleanup and connection management

## Installation

```bash
pip install llmrouter
```

For development:

```bash
pip install llmrouter[dev]
```

## Quick Start

```python
import asyncio
from llmrouter import OpenAIProvider, Message

async def main():
    provider = OpenAIProvider(api_key="your-api-key")
    messages = [Message(role="user", content="Hello!")]
    
    response = await provider.chat(messages)
    print(response.message.content)
    
    await provider.close()

asyncio.run(main())
```

## Detailed Usage

### Provider Configuration

Each provider can be configured with:
- API key (required for OpenAI and Anthropic)
- Base URL (optional, for custom deployments)
- Default model (optional)

```python
# OpenAI with custom configuration
openai = OpenAIProvider(
    api_key="your-api-key",
    base_url="https://api.custom-deployment.com/v1",
    default_model="gpt-4"
)

# Anthropic with default configuration
anthropic = AnthropicProvider(
    api_key="your-api-key"
)

# Ollama for local deployment
ollama = OllamaProvider(
    base_url="http://localhost:11434",
    default_model="llama2"
)
```

### Chat Interface

The chat interface supports conversations with multiple messages:

```python
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What's the weather like?"),
    Message(role="assistant", content="I don't have access to current weather data."),
    Message(role="user", content="What can you help me with?")
]

response = await provider.chat(
    messages=messages,
    temperature=0.7,
    stream=False
)
```

### Streaming Responses

All providers support streaming for both chat and completion endpoints:

```python
async for chunk in await provider.chat(messages, stream=True):
    print(chunk.message.content, end="", flush=True)

async for chunk in await provider.complete(prompt, stream=True):
    print(chunk.text, end="", flush=True)
```

### Error Handling

The library provides consistent error handling across providers:

```python
try:
    response = await provider.chat(messages)
except httpx.HTTPStatusError as e:
    print(f"API error: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Network error: {str(e)}")
finally:
    await provider.close()
```

### Resource Management

Always close providers when done to clean up resources:

```python
try:
    provider = OpenAIProvider(api_key="your-api-key")
    # ... use provider ...
finally:
    await provider.close()
```

Or use async context managers (coming soon):

```python
async with OpenAIProvider(api_key="your-api-key") as provider:
    response = await provider.chat(messages)
```

## Examples

Check out the `examples/` directory for more detailed examples:

- `chat_comparison.py`: Compare responses from different providers
- `streaming_example.py`: Demonstrate streaming capabilities
- `error_handling.py`: Show error handling scenarios

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```
4. Format code:
   ```bash
   black llmrouter
   isort llmrouter
   ```
5. Type check:
   ```bash
   mypy llmrouter
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
