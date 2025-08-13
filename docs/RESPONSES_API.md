# OpenAI Responses API Support

This document describes the new support for OpenAI's Responses API in the `poemai-utils` library.

## Overview

OpenAI has introduced the Responses API as the recommended approach for new applications. This API provides a simpler, more intuitive interface while supporting the same underlying models and capabilities as the Chat Completions API.

## Key Benefits

- **Simpler Interface**: Use `input` instead of complex `messages` arrays
- **Direct Text Output**: Get `output_text` directly instead of nested choice structures
- **Cleaner Parameters**: Use `instructions` instead of system messages
- **Stateful Conversations**: Automatic context management without resending message history
- **Same Models**: All GPT models work with both APIs
- **Lower Costs**: 40-80% cost reduction due to improved cache utilization
- **Better Performance**: 3% improvement in reasoning benchmarks
- **Future-Ready**: Built for OpenAI's upcoming features

## Migration Guide

### Before (Chat Completions API)

```python
from poemai_utils.openai import AskLean

ask = AskLean(openai_api_key="your-key", model="gpt-4o")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

response = ask.ask(messages=messages, max_tokens=100)
answer = response.choices[0].message.content
```

### After (Responses API)

```python
from poemai_utils.openai import AskResponses

ask = AskResponses(openai_api_key="your-key", model="gpt-4o")

response = ask.ask(
    input_data="What is Python?",
    instructions="You are a helpful assistant.",
    max_tokens=100
)
answer = response.output_text
```

### Even Better: Stateful Conversations

```python
from poemai_utils.openai import AskResponses

ask = AskResponses(openai_api_key="your-key", model="gpt-4o")

# Start a stateful conversation - no need to manage message history!
conversation = ask.start_conversation()

# First message
response1 = conversation.send(
    "What is Python?",
    instructions="You are a helpful assistant."
)
print(response1.output_text)

# Follow-up questions automatically maintain context
response2 = conversation.send("What are its main use cases?")
print(response2.output_text)

response3 = conversation.send("How does it compare to Java?")
print(response3.output_text)
```

## API Reference

### AskResponses Class

#### Constructor

```python
AskResponses(
    openai_api_key: str,
    model: str = "gpt-4o",
    base_url: str = "https://api.openai.com/v1/responses",
    timeout: int = 60,
    max_retries: int = 3,
    base_delay: float = 1.0
)
```

#### Methods

##### ask()

The main method for making requests to the Responses API.

```python
ask(
    input_data: Union[str, List[Dict[str, Any]]],
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = 600,
    stop: Optional[Union[str, List[str]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    store: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    include: Optional[List[str]] = None,
    additional_args: Optional[Dict[str, Any]] = None,
) -> Union[PydanticLikeBox, Any]
```

**Parameters:**
- `input_data`: The input to the model (string, list of content objects, or conversation)
- `instructions`: System instructions (replaces system messages)
- `model`: Model to use (overrides instance default)
- `temperature`: Sampling temperature (0-2)
- `max_tokens`: Maximum tokens to generate
- `stop`: Stop sequences
- `tools`: Available tools/functions
- `tool_choice`: Tool choice strategy
- `response_format`: Response format specification
- `stream`: Whether to stream the response
- `store`: Whether to store conversation state (default: True)
- `previous_response_id`: ID of previous response for stateful conversations
- `include`: Additional data to include (e.g., ["reasoning.encrypted_content"])
- `additional_args`: Additional API parameters

##### ask_simple()

Simplified interface for basic text generation.

```python
ask_simple(
    prompt: str,
    instructions: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = 600,
) -> str
```

Returns the generated text directly as a string.

##### ask_vision()

Simplified interface for vision tasks.

```python
ask_vision(
    text: str,
    image_url: str,
    instructions: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = 600,
) -> str
```

**Parameters:**
- `text`: The text prompt
- `image_url`: URL or base64 data URL of the image
- Other parameters as in `ask_simple()`

##### start_conversation()

Create a stateful conversation manager for automatic context management.

```python
start_conversation() -> ConversationManager
```

Returns a `ConversationManager` instance that handles conversation state automatically.

### ConversationManager Class

The `ConversationManager` provides automatic state management for multi-turn conversations.

#### Methods

##### send()

Send a message in the stateful conversation.

```python
send(
    input_data: Union[str, List[Dict[str, Any]]],
    instructions: Optional[str] = None,
    **kwargs
) -> PydanticLikeBox
```

**Parameters:**
- `input_data`: The input message
- `instructions`: System instructions (typically used for first message)
- `**kwargs`: Additional arguments passed to `ask()`

##### reset()

Reset the conversation state to start a new conversation.

```python
reset() -> None
```

##### get_conversation_id()

Get the current conversation ID.

```python
get_conversation_id() -> Optional[str]
```

Returns the ID of the last response, which serves as the conversation identifier.

### Migration Helpers

#### convert_messages_to_input()

Static method to convert Chat Completions messages to Responses API format.

```python
instructions, input_data = AskResponses.convert_messages_to_input(messages)
```

#### AskLean Integration

The existing `AskLean` class now includes methods to use the Responses API:

##### to_responses_api()

Create an `AskResponses` instance with the same configuration.

```python
ask_lean = AskLean(openai_api_key="your-key")
ask_responses = ask_lean.to_responses_api()
```

##### ask_with_responses_api()

Use the Responses API while maintaining Chat Completions interface compatibility.

```python
ask_lean = AskLean(openai_api_key="your-key")

# This uses the Responses API internally but returns Chat Completions format
response = ask_lean.ask_with_responses_api(
    messages=[
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"}
    ]
)

# Access like traditional response
answer = response.choices[0].message.content
```

## Usage Examples

### Stateful Conversations (Recommended)

The most powerful feature of the Responses API is stateful conversations that automatically manage context:

```python
from poemai_utils.openai import AskResponses

ask = AskResponses(openai_api_key="your-key")

# Start a stateful conversation
conversation = ask.start_conversation()

# First message sets the context
response1 = conversation.send(
    "I'm planning a trip to Japan.",
    instructions="You are a helpful travel assistant."
)
print(response1.output_text)

# Follow-up questions maintain full context automatically
response2 = conversation.send("What's the best time to visit?")
print(response2.output_text)

response3 = conversation.send("What about cherry blossom season specifically?")
print(response3.output_text)

response4 = conversation.send("How much should I budget for 2 weeks?")
print(response4.output_text)

# No need to manage message history or previous context!
```

### Basic Text Generation

```python
from poemai_utils.openai import AskResponses

ask = AskResponses(openai_api_key="your-key")

# Simple text generation
response = ask.ask_simple("Explain quantum computing in simple terms.")
print(response)

# With instructions
response = ask.ask_simple(
    prompt="Explain quantum computing",
    instructions="You are a teacher explaining to middle school students."
)
print(response)
```

### Vision Tasks

```python
# Analyze an image
response = ask.ask_vision(
    text="What's in this image?",
    image_url="https://example.com/image.jpg",
    instructions="Describe what you see in detail."
)
print(response)

# With base64 image
import base64
with open("image.png", "rb") as f:
    b64_image = base64.b64encode(f.read()).decode()

response = ask.ask_vision(
    text="What's in this image?",
    image_url=f"data:image/png;base64,{b64_image}"
)
```

### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = ask.ask(
    input_data="What's the weather in Paris?",
    tools=tools,
    tool_choice="auto"
)
```

### Structured Outputs

```python
response = ask.ask(
    input_data="Extract the name and age from: 'John is 25 years old'",
    response_format={"type": "json_object"},
    instructions="Return a JSON object with 'name' and 'age' fields."
)

# Response will be structured JSON
data = json.loads(response.output_text)
print(data["name"], data["age"])
```

### Streaming

```python
stream = ask.ask(
    input_data="Write a short story about a robot.",
    stream=True
)

for chunk in stream:
    if hasattr(chunk, 'delta') and chunk.delta.get('content'):
        print(chunk.delta.content, end='', flush=True)
```

### Multi-turn Conversations (Legacy Approach)

For comparison, here's how multi-turn conversations work without the stateful manager:

```python
# Manual state management (less efficient)
response1 = ask.ask(
    input_data="What is machine learning?",
    instructions="You are a computer science tutor.",
    store=True  # Enable storage to get response IDs
)

print(f"Response 1: {response1.output_text}")
response1_id = response1.id

# Use previous_response_id for context
response2 = ask.ask(
    input_data="What are the main types?",
    previous_response_id=response1_id,
    store=True
)

print(f"Response 2: {response2.output_text}")

# The stateful conversation manager eliminates this manual management
```

### Manual State Management vs ConversationManager

```python
# ❌ Manual approach - verbose and error-prone
messages = []
messages.append({"role": "system", "content": "You are helpful"})
messages.append({"role": "user", "content": "Hello"})

response1 = old_api.ask(messages=messages)
messages.append({"role": "assistant", "content": response1.choices[0].message.content})
messages.append({"role": "user", "content": "Follow-up question"})

response2 = old_api.ask(messages=messages)

# ✅ ConversationManager - clean and automatic
conversation = ask.start_conversation()
response1 = conversation.send("Hello", instructions="You are helpful")
response2 = conversation.send("Follow-up question")
```

### Advanced Stateful Features

#### Zero Data Retention (ZDR) Compliance

For organizations with strict data retention policies:

```python
# ZDR-compliant stateful conversations
response1 = ask.ask(
    input_data="Analyze this sensitive data",
    instructions="You are a data analyst",
    store=False,  # Disable server-side storage
    include=["reasoning.encrypted_content"]  # Use encrypted reasoning
)

# Pass encrypted reasoning to next request
response2 = ask.ask(
    input_data="What are the trends?",
    previous_response_id=None,  # No stored state
    store=False,
    include=["reasoning.encrypted_content"],
    # In practice, you'd pass the encrypted content from response1
)
```

#### Conversation State Introspection

```python
conversation = ask.start_conversation()

response1 = conversation.send("Hello")
print(f"Conversation ID: {conversation.get_conversation_id()}")

response2 = conversation.send("Follow-up")
print(f"Total turns: {len(conversation.conversation_history)}")

# Reset and start fresh
conversation.reset()
print(f"After reset: {conversation.get_conversation_id()}")  # None
```

#### Stateful Conversations with Tools

```python
conversation = ask.start_conversation()

# Tools are automatically available throughout the conversation
response = conversation.send(
    "Search for recent developments in quantum computing",
    instructions="You are a research assistant",
    tools=[{"type": "web_search_preview"}]
)

# Follow-up questions can reference the search results
follow_up = conversation.send("What are the practical applications?")
```

## Error Handling

The `AskResponses` class includes the same robust error handling as `AskLean`:

- Automatic retries for server errors (5xx)
- Exponential backoff
- Configurable timeout and retry limits
- Clear error messages

```python
try:
    response = ask.ask(input_data="Hello")
    print(response.output_text)
except RuntimeError as e:
    print(f"API call failed: {e}")
```

## Model Support

All OpenAI models that support the Chat Completions API also support the Responses API:

- GPT-4 variants (GPT-4o, GPT-4o-mini, etc.)
- GPT-3.5-Turbo variants
- O1 models (o1-preview, o1-mini)
- GPT-5 models (when available)

Use the `OPENAI_MODEL` enum for type safety:

```python
from poemai_utils.openai import AskResponses, OPENAI_MODEL

ask = AskResponses(
    openai_api_key="your-key",
    model=OPENAI_MODEL.GPT_4_o_MINI
)
```

## Testing

The library includes comprehensive tests for the Responses API functionality:

```bash
# Run specific tests
python -m pytest tests/unit/test_ask_responses.py
python -m pytest tests/unit/test_ask_lean_responses.py

# Run all tests
python -m pytest tests/
```

## Backward Compatibility

The existing `AskLean` class and Chat Completions API support are fully maintained. You can:

1. Continue using `AskLean` as before
2. Gradually migrate to `AskResponses`
3. Use the hybrid approach with `ask_with_responses_api()`
4. Mix both APIs in the same application

## Best Practices

1. **New Projects**: Use `AskResponses` for new applications
2. **Existing Projects**: Consider gradual migration or use hybrid methods
3. **Simple Tasks**: Use `ask_simple()` for basic text generation
4. **Complex Tasks**: Use the full `ask()` method with appropriate parameters
5. **Error Handling**: Always wrap API calls in try-catch blocks
6. **Rate Limiting**: Implement application-level rate limiting for production use

## Future Considerations

The Responses API is designed to be the foundation for OpenAI's future features:

- Enhanced structured outputs
- Better function calling
- New model capabilities
- Improved streaming
- Advanced multimodal features

By migrating to the Responses API, you'll be ready for these upcoming enhancements.
