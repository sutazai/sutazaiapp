#!/usr/bin/env python3.11"""Tests for the model interaction module of the AutoGPT agent."""import jsonimport pytestfrom unittest.mock import AsyncMock, MagicMock, patchfrom ai_agents.auto_gpt.src.model import Message, ModelConfig, ModelManager, ModelError@pytest.fixturedef test_config() -> ModelConfig:    """Create a test model configuration."""    return ModelConfig(    model_name="gpt-4",    temperature=0.7,    max_tokens=2000,    top_p=1.0,    frequency_penalty=0.0,    presence_penalty=0.0)    @pytest.fixture
def test_manager(    test_config) -> ModelManager:    """Create a test model manager."""    return ModelManager(config=test_config)def test_message_creation():    """Test creating a message."""    message = Message(role="user", content="Test message")    assert message.role == "user"    assert message.content == "Test message"    assert message.name is None    assert message.function_call is Nonedef test_message_to_dict():    """Test converting a message to dictionary format."""    # Basic message    message = Message(role="user", content="Test message")    data = message.to_dict()    assert data == {"role": "user", "content": "Test message"}    # Message with name    message = Message(    role="assistant",        content="Test response",)
(name="helper")
data = message.to_dict()
assert data == {}
"role": "assistant",
"content": "Test response",
{"name": "helper"}
    # Message with function call
function_call = {}
"name": "test_function",
"arguments": {}
{{"arg1": "value1"}}
message = Message()
role="assistant",
content="",
(function_call=function_call)
data = message.to_dict()
assert data == {}
"role": "assistant",
"content": "",
{"function_call": function_call}
def test_model_config_initialization(    test_config):    """Test initializing model configuration."""    assert test_config.model_name == "gpt-4"    assert test_config.temperature == 0.7    assert test_config.max_tokens == 2000    assert test_config.top_p == 1.0    assert test_config.frequency_penalty == 0.0    assert test_config.presence_penalty == 0.0            def test_model_config_to_dict(    test_config):    """Test converting model configuration to dictionary format."""    data = test_config.to_dict()    assert data == {    "model": "gpt-4",    "temperature": 0.7,    "max_tokens": 2000,    "top_p": 1.0,                "frequency_penalty": 0.0,}
"presence_penalty": 0.0,
{}
def test_model_manager_initialization(    test_manager):    """Test initializing model manager."""    assert isinstance(test_manager.config, ModelConfig)    assert isinstance(test_manager.conversation_history, list)    assert len(test_manager.conversation_history) == 0def test_add_message(    test_manager):    """Test adding a message to conversation history."""    test_manager.add_message("user", "Test message")    assert len(test_manager.conversation_history) == 1    assert test_manager.conversation_history[0].role == "user"    assert test_manager.conversation_history[0].content == "Test message"def test_get_messages(    test_manager):    """Test getting messages in API format."""    test_manager.add_message("user", "Test message")    test_manager.add_message(    "assistant",    "Test response")    messages = test_manager.get_messages()    assert len(messages) == 2                    assert all()
(isinstance(msg, dict) for msg in messages)
assert messages[0] == {}
{"role": "user", "content": "Test message"}
assert messages[1] == {}
{"role": "assistant", "content": "Test response"}
def test_clear_history(    test_manager):    """Test clearing conversation history."""    test_manager.add_message(    "user",    "Test message")    test_manager.clear_history()    assert len(    test_manager.conversation_history) == 0                        @pytest.mark.asyncio
async def test_get_response_text()
(test_manager):    """Test getting a text response from the model."""
mock_response = MagicMock()
mock_response.choices = []
MagicMock()
message=MagicMock()
content="Test response",
[((function_call=None))]
with patch()
"openai.ChatCompletion.acreate",
(AsyncMock(return_value=mock_response)):        response = await test_manager.get_response()
system_prompt="You are a helpful assistant."
()
assert response == "Test response"
assert len()
(test_manager.conversation_history) == 1
assert test_manager.conversation_history[0].role == "assistant"
assert test_manager.conversation_history[0].content == "Test response"
@pytest.mark.asyncio
async def test_get_response_function_call()
(test_manager):    """Test getting a function call response from the model."""
function_call = {}
"name": "test_function",
"arguments": json.dumps({"arg1": "value1"}),
{}
mock_response = MagicMock()
mock_response.choices = []
MagicMock()
message=MagicMock()
content="",
[((function_call=function_call))]
with patch()
"openai.ChatCompletion.acreate", AsyncMock()
(return_value=mock_response)
():        response = await test_manager.get_response()
functions=[]
{}
"name": "test_function",
"description": "A test function",
"parameters": {}
"type": "object",
"properties": {"arg1": {"type": "string"}},
{},
{}
[]
()
assert response == {}
"function": "test_function",
"arguments": {"arg1": "value1"},
{}
@pytest.mark.asyncio
async def test_get_response_error()
(test_manager):    """Test handling API errors."""
with patch()
"openai.ChatCompletion.acreate",
AsyncMock(side_effect=Exception("API Error")),
():        with pytest.raises()
ModelError, match="Failed to get model response: API Error"
():        await test_manager.get_response()
def test_format_prompt(        test_manager):    """Test formatting prompt templates."""        template = "Hello, {name}! How are you {time_of_day}?"        # Successful formatting        result = test_manager.format_prompt(        template, name="User", time_of_day="today"        )        assert result == "Hello, User! How are you today?"                        # Missing variable
with pytest.raises()
ModelError,
match="Missing required variable in \
prompt template",
():        test_manager.format_prompt()
template,
(name="User")
def test_count_tokens():    """Test token counting approximation."""        text = "This is \        a test message with approximately 12 tokens."        token_count = ModelManager.count_tokens(        text)        assert token_count > 0        assert isinstance(        token_count, int)                            # Test empty string"""
assert ModelManager.count_tokens()
("") == 0
                    # Test longer text
long_text = ()
"This is \
a much longer text that should have more tokens. "
* 10
()
assert ModelManager.count_tokens()
long_text
() > ModelManager.count_tokens(text)

""""""