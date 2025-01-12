"""Example demonstrating Anthropic streaming functionality."""
import asyncio
import os
from llmrouter import AnthropicProvider, Message

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
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        default_model="claude-3-opus-20240229"
    )
    
    try:
        # Stream chat response
        messages = [
            Message(role="system", content="You are a science educator."),
            Message(role="user", content="Explain how black holes work and their impact on space-time.")
        ]
        await stream_chat(provider, messages)
        
        # Stream chat with different parameters
        messages = [Message(role="user", content="Write an essay about artificial intelligence.")]
        await stream_chat(
            provider,
            messages,
            temperature=0.3,
            model="claude-3-sonnet-20240229"
        )
        
        # Stream completion
        prompt = "Write a detailed analysis of Shakespeare's Hamlet:"
        await stream_completion(provider, prompt)
        
        # Stream completion with parameters
        await stream_completion(
            provider,
            "Explain the process of neural network training:",
            temperature=0.7,
            max_tokens=500
        )
        
        # Stream a long response to see chunks
        print("\nStreaming a long response to demonstrate chunking:")
        messages = [Message(
            role="user",
            content="Write a comprehensive guide to modern cryptography, "
                   "including symmetric and asymmetric encryption, hashing, "
                   "and digital signatures."
        )]
        await stream_chat(provider, messages)
        
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(main()) 