"""Example comparing responses from different LLM providers."""
import asyncio
import os
from simplemodelrouter import OpenAIProvider, AnthropicProvider, OllamaProvider, Message

async def get_response(provider, prompt: str) -> str:
    """Get a response from a provider."""
    messages = [Message(role="user", content=prompt)]
    response = await provider.chat(messages)
    return response.message.content

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
    
    # Test prompt
    prompt = "Explain the concept of quantum entanglement in simple terms."
    
    try:
        # Get responses from all providers
        print("Getting responses from all providers...\n")
        
        print("OpenAI response:")
        openai_response = await get_response(openai, prompt)
        print(f"{openai_response}\n")
        
        print("Anthropic response:")
        anthropic_response = await get_response(anthropic, prompt)
        print(f"{anthropic_response}\n")
        
        print("Ollama response:")
        ollama_response = await get_response(ollama, prompt)
        print(f"{ollama_response}\n")
        
    finally:
        # Clean up
        await asyncio.gather(
            openai.close(),
            anthropic.close(),
            ollama.close()
        )

if __name__ == "__main__":
    asyncio.run(main()) 