"""Example demonstrating OpenAI streaming functionality."""
import asyncio
import os
from llmrouter import OpenAIProvider, Message

async def stream_chat(provider, messages, **kwargs):
    """Stream a chat response."""
    print("\nStreaming chat response:")
    async for chunk in await provider.chat(messages, stream=True, **kwargs):
        print(chunk.message.content, end="", flush=True)
    print("\n")

async def stream_completion(provider, prompt, **kwargs):
    """Stream a completion response."""
    print("\nStreaming completion response:")
    async for chunk in await provider.complete(prompt, stream=True, **kwargs):
        print(chunk.text, end="", flush=True)
    print("\n")

async def main():
    # Initialize provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="gpt-3.5-turbo"
    )
    
    try:
        # Stream chat response
        messages = [
            Message(role="system", content="You are a storyteller."),
            Message(role="user", content="Tell me a story about a magical library.")
        ]
        await stream_chat(provider, messages)
        
        # Stream chat with different parameters
        messages = [Message(role="user", content="Write a poem about the ocean.")]
        await stream_chat(
            provider,
            messages,
            temperature=0.9,
            model="gpt-4"
        )
        
        # Stream completion
        prompt = "Write a step-by-step guide to making the perfect cup of coffee:"
        await stream_completion(provider, prompt)
        
        # Stream completion with parameters
        await stream_completion(
            provider,
            "Explain quantum computing in simple terms:",
            temperature=0.7,
            max_tokens=200
        )
        
        # Stream a long response to see chunks
        print("\nStreaming a long response to demonstrate chunking:")
        messages = [Message(
            role="user",
            content="Write a detailed explanation of how the internet works, "
                   "including protocols, infrastructure, and data transmission."
        )]
        await stream_chat(provider, messages)
        
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(main()) 