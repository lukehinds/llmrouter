"""Example demonstrating OpenAI completion functionality."""
import asyncio
import os
from llmrouter import OpenAIProvider

async def main():
    # Initialize provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="text-davinci-003"  # Completion model
    )
    
    try:
        # Simple completion
        print("Simple completion example:")
        response = await provider.complete("The capital of France is")
        print(f"Response: {response.text}\n")
        
        # Completion with parameters
        print("Completion with parameters:")
        response = await provider.complete(
            prompt="Write a haiku about",
            temperature=0.9,  # More creative
            max_tokens=50  # Limit response length
        )
        print(f"Response: {response.text}\n")
        
        # Multiple completions
        print("Multiple completions:")
        prompts = [
            "Write a function that",
            "Create a recipe for",
            "The best way to learn is"
        ]
        
        for prompt in prompts:
            response = await provider.complete(prompt)
            print(f"Prompt: {prompt}")
            print(f"Response: {response.text}\n")
            
        # Print token usage for last completion
        print("Token usage:")
        print(f"Prompt tokens: {response.usage['prompt_tokens']}")
        print(f"Completion tokens: {response.usage['completion_tokens']}")
        print(f"Total tokens: {response.usage['total_tokens']}")
        
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(main()) 