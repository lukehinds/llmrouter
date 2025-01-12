"""Example demonstrating Ollama completion functionality.

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
from simplemodelrouter import OllamaProvider

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
        # Simple completion
        print("Simple completion example:")
        try:
            response = await provider.complete("Write a Python function to sort a list:")
            print(f"Response: {response.text}\n")
        except httpx.HTTPError as e:
            print(f"Error during completion request: {str(e)}")
            return
        
        # Test different parameters
        print("Testing different parameters:")
        parameters = [
            {"temperature": 0.1, "max_tokens": 100},  # More focused, shorter
            {"temperature": 0.9, "max_tokens": 200},  # More creative, longer
            {"temperature": 0.5, "max_tokens": 150}   # Balanced
        ]
        
        prompt = "Explain the concept of machine learning:"
        for params in parameters:
            try:
                print(f"\nUsing parameters: {params}")
                response = await provider.complete(prompt, **params)
                print(f"Response: {response.text}")
            except httpx.HTTPError as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue
        
        # Try different available models
        print("\nTrying different models for code completion:")
        code_prompts = [
            "Write a function to implement binary search:",
            "Create a class for a linked list:",
            "Implement a simple web server:"
        ]
        
        for prompt, model in zip(code_prompts, available_models[:3]):
            try:
                print(f"\nUsing model: {model}")
                print(f"Prompt: {prompt}")
                response = await provider.complete(prompt, model=model)
                print(f"Response: {response.text}\n")
            except Exception as e:
                print(f"Error with model {model}: {str(e)}")
                continue
        
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