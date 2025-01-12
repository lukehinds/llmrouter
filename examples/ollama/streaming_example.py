"""Example demonstrating Ollama streaming functionality.

Before running this example:
1. Install Ollama from https://ollama.ai
2. Start the Ollama service
3. Pull required models using:
   - ollama pull llama2
   - ollama pull mistral
   - ollama pull codellama
"""
import asyncio
import sys
import httpx
from simplemodelrouter import OllamaProvider, Message

async def check_ollama_service(base_url: str) -> bool:
    """Check if Ollama service is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/tags")
            return response.status_code == 200
    except Exception:
        return False

async def list_available_models(base_url: str) -> list:
    """List available models in Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json()["models"]]
    except Exception:
        return []
    return []

async def stream_chat(provider, messages, **kwargs):
    """Stream a chat response."""
    print("\nStreaming chat response:")
    try:
        async for chunk in await provider.chat(messages, stream=True, **kwargs):
            print(chunk.message.content, end="", flush=True)
        print("\n")
    except httpx.HTTPError as e:
        print(f"\nError during chat streaming: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error during chat streaming: {str(e)}")

async def stream_completion(provider, prompt, **kwargs):
    """Stream a completion response."""
    print("\nStreaming completion response:")
    try:
        async for chunk in await provider.complete(prompt, stream=True, **kwargs):
            print(chunk.text, end="", flush=True)
        print("\n")
    except httpx.HTTPError as e:
        print(f"\nError during completion streaming: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error during completion streaming: {str(e)}")

async def main():
    base_url = "http://localhost:11434"
    
    # Check if Ollama service is running
    if not await check_ollama_service(base_url):
        print("Error: Ollama service is not running. Please:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Start the Ollama service")
        sys.exit(1)
    
    # List available models
    available_models = await list_available_models(base_url)
    if not available_models:
        print("Warning: No models found. Please pull models using:")
        print("- ollama pull llama2")
        print("- ollama pull mistral")
        print("- ollama pull codellama")
        sys.exit(1)
    
    print(f"Available models: {', '.join(available_models)}\n")
    
    # Initialize provider with first available model
    default_model = available_models[0] if available_models else "llama2"
    provider = OllamaProvider(
        base_url=base_url,
        default_model=default_model
    )
    
    try:
        # Stream chat response with different models
        for model in available_models[:3]:  # Try first 3 available models
            print(f"\nStreaming chat with model: {model}")
            messages = [
                Message(role="system", content="You are a helpful coding assistant."),
                Message(role="user", content="Write a Python class for a binary search tree.")
            ]
            await stream_chat(provider, messages, model=model)
        
        # Stream chat with different parameters
        print("\nStreaming with different parameters:")
        messages = [Message(role="user", content="Write a technical blog post about microservices.")]
        await stream_chat(
            provider,
            messages,
            temperature=0.8,
            model=default_model
        )
        
        # Stream completion with code generation
        if "codellama" in available_models:
            print("\nStreaming code completion with CodeLlama:")
            await stream_completion(
                provider,
                "Write a FastAPI application with the following endpoints:",
                model="codellama"
            )
        
        # Stream completion with different contexts
        prompts = [
            "Explain the process of photosynthesis step by step:",
            "Write a creative story about time travel:",
            "Describe the architecture of a modern web application:"
        ]
        
        for prompt in prompts:
            print(f"\nStreaming response for: {prompt}")
            await stream_completion(provider, prompt, model=default_model)
        
        # Stream a long technical explanation
        print("\nStreaming a long technical explanation:")
        messages = [Message(
            role="user",
            content="Provide a comprehensive explanation of how blockchain technology works, "
                   "including concepts like consensus mechanisms, smart contracts, and "
                   "cryptographic principles."
        )]
        await stream_chat(provider, messages, model=default_model)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        await provider.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 