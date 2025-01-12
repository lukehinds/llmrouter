"""Example demonstrating streaming responses from different LLM providers."""
import asyncio
import os
from modelrouter import OpenAIProvider, AnthropicProvider, OllamaProvider, Message

async def stream_chat(provider, prompt: str):
    """Stream a chat response from a provider."""
    messages = [Message(role="user", content=prompt)]
    
    print(f"\nStreaming response from {provider.__class__.__name__}:")
    async for chunk in await provider.chat(messages, stream=True):
        print(chunk.message.content, end="", flush=True)
    print("\n")

async def stream_completion(provider, prompt: str):
    """Stream a completion response from a provider."""
    print(f"\nStreaming completion from {provider.__class__.__name__}:")
    async for chunk in await provider.complete(prompt, stream=True):
        print(chunk.text, end="", flush=True)
    print("\n")

async def main():
    # Initialize providers
    openai = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="gpt-3.5-turbo"
    )
    
    anthropic = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        default_model="claude-3-opus-20240229"
    )
    
    ollama = OllamaProvider(
        base_url="http://localhost:11434",
        default_model="llama2"
    )
    
    # Test prompts
    chat_prompt = "Write a short story about a time traveler."
    completion_prompt = "Explain how a bicycle works."
    
    try:
        # Stream chat responses
        print("Streaming chat responses...")
        await stream_chat(openai, chat_prompt)
        await stream_chat(anthropic, chat_prompt)
        await stream_chat(ollama, chat_prompt)
        
        # Stream completion responses
        print("\nStreaming completion responses...")
        await stream_completion(openai, completion_prompt)
        await stream_completion(anthropic, completion_prompt)
        await stream_completion(ollama, completion_prompt)
        
    finally:
        # Clean up
        await asyncio.gather(
            openai.close(),
            anthropic.close(),
            ollama.close()
        )

if __name__ == "__main__":
    asyncio.run(main()) 