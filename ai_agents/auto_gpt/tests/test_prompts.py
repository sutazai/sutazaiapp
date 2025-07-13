#!/usr/bin/env python3.11
"""Tests for the prompt management module of the AutoGPT agent."""

import pytest
from datetime import datetime

    PromptTemplate,
    PromptVariable,
    PromptManager,
    PromptError,
)


def test_prompt_variable_creation():
    """Test creating a prompt variable."""
    var = PromptVariable(
        name="test_var",
        description="Test variable",
        required=True,
        default_value="default",
    )
    assert var.name == "test_var"
    assert var.description == "Test variable"
    assert var.required is True
    assert var.default_value == "default"


def test_prompt_variable_validation():
    """Test prompt variable validation."""
    var = PromptVariable(
        name="test_var",
        description="Test variable",
        required=True,
    )

    # Test required variable without value
    with pytest.raises(PromptError):
        var.validate(None)

    # Test required variable with value
    assert var.validate("test value") == "test value"

    # Test optional variable without value
    var.required = False
    assert var.validate(None) is None

    # Test with default value
    var.default_value = "default"
    assert var.validate(None) == "default"


def test_prompt_template_creation():
    """Test creating a prompt template."""
    template = PromptTemplate(
        name="test_template",
        description="Test template",
        template_text="Hello {name}!",
        variables=[
            PromptVariable(
                name="name",
                description="Name variable",
                required=True,
            ),
        ],
    )
    assert template.name == "test_template"
    assert template.description == "Test template"
    assert template.template_text == "Hello {name}!"
    assert len(template.variables) == 1
    assert template.variables[0].name == "name"


def test_prompt_template_format():
    """Test formatting a prompt template."""
    template = PromptTemplate(
        name="test_template",
        description="Test template",
        template_text="Hello {name}! Today is {date}.",
        variables=[
            PromptVariable(
                name="name",
                description="Name variable",
                required=True,
            ),
            PromptVariable(
                name="date",
                description="Date variable",
                required=True,
            ),
        ],
    )

    # Test with all variables
    result = template.format({"name": "John", "date": "Monday"})
    assert result == "Hello John! Today is Monday."

    # Test with missing required variable
    with pytest.raises(PromptError):
        template.format({"name": "John"})

    # Test with extra variables
    result = template.format({
        "name": "John",
        "date": "Monday",
        "extra": "ignored",
    })
    assert result == "Hello John! Today is Monday."


def test_prompt_manager():
    """Test prompt manager functionality."""
    manager = PromptManager()

    # Test template registration
    template = PromptTemplate(
        name="test_template",
        description="Test template",
        template_text="Hello {name}!",
        variables=[
            PromptVariable(
                name="name",
                description="Name variable",
                required=True,
            ),
        ],
    )
    manager.register_template(template)

    # Test template retrieval
    retrieved = manager.get_template("test_template")
    assert retrieved.name == "test_template"

    # Test non-existent template
    with pytest.raises(PromptError):
        manager.get_template("non_existent")

    # Test template listing
    templates = manager.list_templates()
    assert len(templates) == 1
    assert templates[0].name == "test_template"

    # Test template unregistration
    manager.unregister_template("test_template")
    with pytest.raises(PromptError):
        manager.get_template("test_template")


def test_prompt_manager_format():
    """Test formatting prompts through the manager."""
    manager = PromptManager()

    template = PromptTemplate(
        name="greeting",
        description="Greeting template",
        template_text="Hello {name}! The time is {time}.",
        variables=[
            PromptVariable(
                name="name",
                description="Name variable",
                required=True,
            ),
            PromptVariable(
                name="time",
                description="Time variable",
                required=True,
            ),
        ],
    )
    manager.register_template(template)

    # Test successful formatting
    result = manager.format_prompt(
        "greeting",
        {
            "name": "John",
            "time": datetime.now().strftime("%H:%M"),
        },
    )
    assert "Hello John!" in result
    assert "The time is" in result

    # Test missing variables
    with pytest.raises(PromptError):
        manager.format_prompt("greeting", {"name": "John"})

    # Test non-existent template
    with pytest.raises(PromptError):
        manager.format_prompt("non_existent", {})
