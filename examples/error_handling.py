"""Example demonstrating error handling with LLM providers."""
import asyncio
import httpx
from llmrouter import OpenAIProvider, AnthropicProvider, OllamaProvider, Message

async def test_provider(provider, name: str):
    """Test a provider with various error scenarios."""
    print(f"\nTesting {name}...")
    
    try:
        # Test with invalid API key
        provider.api_key = "invalid-key"
        messages = [Message(role="user", content="Hello")]
        
        try:
            await provider.chat(messages)
        except httpx.HTTPStatusError as e:
            print(f"✓ Caught authentication error as expected: {e.response.status_code}")
            
        # Test with invalid model
        provider.default_model = "non-existent-model"
        try:
            await provider.chat(messages)
        except httpx.HTTPStatusError as e:
            print(f"✓ Caught invalid model error as expected: {e.response.status_code}")
            
        # Test with invalid base URL
        provider.base_url = "https://invalid-url.example.com"
        try:
            await provider.chat(messages)
        except httpx.RequestError as e:
            print(f"✓ Caught connection error as expected: {str(e)}")
            
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
    finally:
        await provider.close()

async def main():
    # Test OpenAI provider
    openai = OpenAIProvider(api_key="test")
    await test_provider(openai, "OpenAI")
    
    # Test Anthropic provider
    anthropic = AnthropicProvider(api_key="test")
    await test_provider(anthropic, "Anthropic")
    
    # Test Ollama provider
    ollama = OllamaProvider()
    await test_provider(ollama, "Ollama")

if __name__ == "__main__":
    asyncio.run(main()) 