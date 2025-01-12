"""Example demonstrating OpenAI chat functionality."""
import asyncio
import os
from llmrouter import OpenAIProvider, Message

async def main():
    # Initialize provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="gpt-3.5-turbo"  # You can also use "gpt-4" if you have access
    )
    
    try:
        # Single message chat
        print("Simple chat example:")
        messages = [Message(role="user", content="What is Python?")]
        response = await provider.chat(messages)
        print(f"Response: {response.message.content}\n")
        
        # Multi-message conversation
        print("Conversation example:")
        messages = [
            Message(role="system", content="You are a Python programming expert."),
            Message(role="user", content="What's the difference between a list and a tuple?"),
            Message(role="assistant", content="Lists are mutable sequences while tuples are immutable."),
            Message(role="user", content="Can you give me an example?")
        ]
        response = await provider.chat(messages)
        print(f"Response: {response.message.content}\n")
        
        # Using different parameters
        print("Using different parameters:")
        messages = [Message(role="user", content="Write a short poem about coding.")]
        response = await provider.chat(
            messages,
            temperature=0.9,  # More creative
            model="gpt-4"  # Using a different model
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