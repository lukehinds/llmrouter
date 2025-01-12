"""Example demonstrating Anthropic completion functionality."""
import asyncio
import os
from llmrouter import AnthropicProvider

async def main():
    # Initialize provider
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        default_model="claude-3-opus-20240229"
    )
    
    try:
        # Simple completion
        print("Simple completion example:")
        response = await provider.complete("Write a function to calculate the Fibonacci sequence:")
        print(f"Response: {response.text}\n")
        
        # Completion with parameters
        print("Completion with parameters:")
        response = await provider.complete(
            prompt="Explain the theory of relativity in simple terms:",
            temperature=0.3,  # More focused
            max_tokens=300  # Limit response length
        )
        print(f"Response: {response.text}\n")
        
        # Multiple completions
        print("Multiple completions:")
        prompts = [
            "Write a SQL query to",
            "Explain how photosynthesis works:",
            "What are the key principles of machine learning?"
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