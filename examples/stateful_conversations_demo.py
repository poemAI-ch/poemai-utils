#!/usr/bin/env python3
"""
Example script demonstrating stateful conversations with the OpenAI Responses API.

This shows how to use the new stateful conversation features that eliminate the
need to manually manage and send message history.
"""

import os

from poemai_utils.openai import AskResponses


def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("=== OpenAI Responses API Stateful Conversations ===\n")
    
    # Create the AskResponses instance
    ask = AskResponses(openai_api_key=api_key, model="gpt-4o-mini")
    
    print("1. Traditional Multi-turn (Manual State Management)")
    print("-" * 50)
    
    # Traditional approach: Manual state management
    response1 = ask.ask(
        input_data="What is the capital of France?",
        instructions="You are a helpful geography assistant.",
        store=True  # Enable storage to get response IDs
    )
    
    print(f"Q: What is the capital of France?")
    print(f"A: {response1.output_text}")
    print(f"Response ID: {getattr(response1, 'id', 'N/A')}")
    
    # Follow-up question using previous_response_id
    response2 = ask.ask(
        input_data="What about its population?",
        previous_response_id=getattr(response1, 'id', None),
        store=True
    )
    
    print(f"\nQ: What about its population?")
    print(f"A: {response2.output_text}")
    print(f"Response ID: {getattr(response2, 'id', 'N/A')}")
    
    print("\n" + "="*70 + "\n")
    
    print("2. Stateful Conversation Manager (Automatic State Management)")
    print("-" * 60)
    
    # New approach: Automatic state management with ConversationManager
    conversation = ask.start_conversation()
    
    # First message
    response1 = conversation.send(
        input_data="Tell me about renewable energy.",
        instructions="You are an expert on environmental topics."
    )
    
    print(f"Q: Tell me about renewable energy.")
    print(f"A: {response1.output_text}")
    print(f"Conversation ID: {conversation.get_conversation_id()}")
    
    # Follow-up questions - no need to manage state manually
    response2 = conversation.send("What are the main types?")
    print(f"\nQ: What are the main types?")
    print(f"A: {response2.output_text}")
    
    response3 = conversation.send("Which is most cost-effective?")
    print(f"\nQ: Which is most cost-effective?")
    print(f"A: {response3.output_text}")
    
    response4 = conversation.send("How does this compare to fossil fuels in terms of job creation?")
    print(f"\nQ: How does this compare to fossil fuels in terms of job creation?")
    print(f"A: {response4.output_text}")
    
    print(f"\nFinal Conversation ID: {conversation.get_conversation_id()}")
    print(f"Conversation history length: {len(conversation.conversation_history)}")
    
    print("\n" + "="*70 + "\n")
    
    print("3. Benefits of Stateful Conversations")
    print("-" * 35)
    print("✓ No need to manage message arrays")
    print("✓ Automatic context preservation") 
    print("✓ Lower costs due to improved cache utilization")
    print("✓ Better reasoning performance")
    print("✓ Simplified code for multi-turn interactions")
    print("✓ Server-side conversation state management")
    
    print("\n" + "="*70 + "\n")
    
    print("4. Advanced Features")
    print("-" * 20)
    
    # Demonstrate tools with stateful conversations
    print("Using tools in stateful conversations:")
    
    tools_conversation = ask.start_conversation()
    
    try:
        response = tools_conversation.send(
            input_data="Search for the latest news about artificial intelligence breakthroughs.",
            instructions="You are a research assistant. Use web search when needed.",
            tools=[{"type": "web_search_preview"}]  # Built-in web search tool
        )
        
        print(f"Q: Search for the latest news about artificial intelligence breakthroughs.")
        print(f"A: {response.output_text}")
        
        # Follow-up without re-specifying tools or context
        followup = tools_conversation.send("What are the implications of these breakthroughs?")
        print(f"\nQ: What are the implications of these breakthroughs?")
        print(f"A: {followup.output_text}")
        
    except Exception as e:
        print(f"Tool usage example failed (this is expected without proper API access): {e}")
    
    print("\n" + "="*70 + "\n")
    
    print("5. Zero Data Retention (ZDR) Support")
    print("-" * 35)
    print("For organizations with compliance requirements:")
    
    # Example of ZDR-compatible usage
    try:
        zdr_response = ask.ask(
            input_data="Analyze this sensitive data...",
            instructions="You are a data analyst.",
            store=False,  # Disable storage for ZDR compliance
            include=["reasoning.encrypted_content"]  # Use encrypted reasoning
        )
        
        print("✓ ZDR-compliant request made with encrypted reasoning")
        print("✓ No conversation state persisted on servers")
        print("✓ Reasoning tokens encrypted and returned for local management")
        
    except Exception as e:
        print(f"ZDR example (expected to work with proper setup): {e}")

if __name__ == "__main__":
    main()
