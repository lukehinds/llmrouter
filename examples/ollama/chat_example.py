"""Example demonstrating Ollama chat functionality.

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
from llmrouter import OllamaProvider, Message

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
        # Single message chat
        print("Simple chat example:")
        messages = [Message(role="user", content="What are the benefits of open-source software?")]
        try:
            response = await provider.chat(messages)
            print(f"Response: {response.message.content}\n")
        except httpx.HTTPError as e:
            print(f"Error during chat request: {str(e)}")
            return
        
        # Multi-message conversation
        print("Conversation example:")
        messages = [
            Message(role="system", content="You are an expert in computer science."),
            Message(role="user", content="What is a binary tree?"),
            Message(role="assistant", content="A binary tree is a hierarchical data structure where each node has at most two children."),
            Message(role="user", content="What are its common applications?")
        ]
        try:
            response = await provider.chat(messages)
            print(f"Response: {response.message.content}\n")
        except httpx.HTTPError as e:
            print(f"Error during conversation: {str(e)}")
            return
        
        # Try different available models
        print("\nTrying different models:")
        for model in available_models[:3]:  # Try first 3 available models
            try:
                print(f"\nUsing model: {model}")
                messages = [Message(role="user", content="What is your name and what can you do?")]
                response = await provider.chat(messages, model=model)
                print(f"Response: {response.message.content}")
            except Exception as e:
                print(f"Error with model {model}: {str(e)}")
        
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