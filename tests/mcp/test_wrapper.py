#!/usr/bin/env python3
"""
Tests for the MCP wrapper
"""

import pytest
import json
from unittest. import patch, Magic

from task_runner.mcp.wrapper import format_response


def test_format_response():
    """Test formatting MCP responses"""
    # Test success response
    success_response = format_response(True, data={"result": "success"})
    assert success_response["success"] is True
    assert success_response["result"] == "success"
    
    # Test error response
    error_response = format_response(False, error="Error message")
    assert error_response["success"] is False
    assert error_response["error"] == "Error message"
    
    # Test empty responses
    empty_success = format_response(True)
    assert empty_success["success"] is True
    assert len(empty_success) == 1
    
    empty_error = format_response(False)
    assert empty_error["success"] is False
    assert len(empty_error) == 1


# For actual MCP wrapper tests, we would need to  FastMCP
# These are basic example tests that don't require FastMCP to be installed
def test_handler_response_format():
    """Test that handler responses have the correct format"""
    # Import handlers directly to avoid FastMCP dependency
    from task_runner.mcp.wrapper import (
        run_task_handler,
        run_all_tasks_handler,
        parse_task_list_handler,
        create_project_handler,
        get_task_status_handler,
        get_task_summary_handler,
        clean_handler
    )
    
    #  the TaskManager
    with patch("task_runner.mcp.wrapper.TaskManager") as TaskManager:
        # Configure the 
        _manager = Magic()
        TaskManager.return_value = _manager
        
        # Test error responses when required parameters are missing
        response = run_task_handler({})
        assert response["success"] is False
        assert "error" in response
        
        response = parse_task_list_handler({})
        assert response["success"] is False
        assert "error" in response
        
        response = create_project_handler({})
        assert response["success"] is False
        assert "error" in response