#!/usr/bin/env python3
"""
Example script demonstrating how to use both the Chat Completions API (AskLean) 
and the new Responses API (AskResponses).

This shows migration patterns and usage examples.
"""

import os
from poemai_utils.openai import AskLean, AskResponses

def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("=== OpenAI API Usage Examples ===\n")
    
    # Example 1: Traditional Chat Completions API
    print("1. Traditional Chat Completions API (AskLean)")
    ask_lean = AskLean(openai_api_key=api_key, model="gpt-4o-mini")
    
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is the difference between a list and a tuple in Python?"}
    ]
    
    try:
        response = ask_lean.ask(messages=messages, max_tokens=200)
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Error with Chat Completions API: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: New Responses API
    print("2. New Responses API (AskResponses)")
    ask_responses = AskResponses(openai_api_key=api_key, model="gpt-4o-mini")
    
    try:
        response = ask_responses.ask(
            input_data="What is the difference between a list and a tuple in Python?",
            instructions="You are a helpful coding assistant.",
            max_tokens=200
        )
        print(f"Response: {response.output_text}")
        print(f"Model: {response.model}")
        print(f"Usage: {getattr(response, 'usage', 'Not available')}")
    except Exception as e:
        print(f"Error with Responses API: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Simplified interface
    print("3. Simplified Responses API interface")
    try:
        simple_response = ask_responses.ask_simple(
            prompt="Explain recursion in one sentence.",
            instructions="Be concise and clear."
        )
        print(f"Simple response: {simple_response}")
    except Exception as e:
        print(f"Error with simple interface: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Migration helper
    print("4. Using migration helper in AskLean")
    try:
        # Use the migration helper method
        response = ask_lean.ask_with_responses_api(
            messages=messages,
            max_tokens=200
        )
        print(f"Migrated response: {response.choices[0].message.content}")
        print("(This was actually processed using the Responses API but returned in Chat Completions format)")
    except Exception as e:
        print(f"Error with migration helper: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Converting messages format
    print("5. Converting message format")
    instructions, input_data = AskResponses.convert_messages_to_input(messages)
    print(f"Instructions: {instructions}")
    print(f"Input data: {input_data}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 6: Vision example (requires image URL)
    print("6. Vision example (commented out - requires image URL)")
    print("""
    # Example with vision
    try:
        vision_response = ask_responses.ask_vision(
            text="What's in this image?",
            image_url="https://example.com/image.jpg",
            instructions="Describe what you see in detail."
        )
        print(f"Vision response: {vision_response}")
    except Exception as e:
        print(f"Error with vision: {e}")
    """)
    
    print("\n=== Migration Benefits ===")
    print("✓ Simpler API interface")
    print("✓ Direct text output")
    print("✓ Cleaner parameter names")
    print("✓ Same models and capabilities")
    print("✓ Better structured for new features")

if __name__ == "__main__":
    main()
