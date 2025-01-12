"""Example demonstrating Anthropic chat functionality."""
import asyncio
import os
from llmrouter import AnthropicProvider, Message

async def main():
    # Initialize provider
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        default_model="claude-3-opus-20240229"
    )
    
    try:
        # Single message chat
        print("Simple chat example:")
        messages = [Message(role="user", content="Explain the concept of recursion.")]
        response = await provider.chat(messages)
        print(f"Response: {response.message.content}\n")
        
        # Multi-message conversation
        print("Conversation example:")
        messages = [
            Message(role="system", content="You are a mathematics tutor."),
            Message(role="user", content="What is calculus used for?"),
            Message(role="assistant", content="Calculus is used to study rates of change and accumulation."),
            Message(role="user", content="Can you give me a real-world example?")
        ]
        response = await provider.chat(messages)
        print(f"Response: {response.message.content}\n")
        
        # Using different parameters
        print("Using different parameters:")
        messages = [Message(role="user", content="Explain quantum mechanics to a 5-year-old.")]
        response = await provider.chat(
            messages,
            temperature=0.5,  # More focused
            model="claude-3-sonnet-20240229"  # Using a different model
        )
        print(f"Response: {response.message.content}\n")
        
        # Print token usage
        print("Token usage:")
        print(f"Prompt tokens: {response.usage['prompt_tokens']}")
        print(f"Completion tokens: {response.usage['completion_tokens']}")
        print(f"Total tokens: {response.usage['total_tokens']}")
        
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(main()) 