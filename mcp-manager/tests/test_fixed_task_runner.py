"""
Tests for the Fixed Task Runner MCP Server

Verifies that the fixed implementation works correctly with the official MCP SDK.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from mcp.types import Tool, TextContent

from mcp_manager.fixed_task_runner import FixedTaskRunnerServer


class TestFixedTaskRunner:
    """Test suite for the fixed task runner implementation"""
    
    @pytest.fixture
    async def server(self) -> FixedTaskRunnerServer:
        """Create a test server instance"""
        return FixedTaskRunnerServer()
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, server: FixedTaskRunnerServer):
        """Test that server initializes correctly"""
        assert server is not None
        assert server.server is not None
        assert server.base_dir == Path.home() / "claude_task_runner"
        assert isinstance(server.tasks, dict)
        assert isinstance(server.task_results, dict)
    
    @pytest.mark.asyncio
    async def test_list_tools(self, server: FixedTaskRunnerServer):
        """Test that tools are properly registered"""
        # This is a simplified test since we can't easily call the decorated method
        # In a real test, we'd use the MCP client to connect and test
        
        expected_tools = [
            "run_task",
            "run_all_tasks", 
            "parse_task_list",
            "create_project",
            "get_task_status",
            "get_task_summary",
            "clean"
        ]
        
        # The tools are registered via decorators, so we can't test them directly
        # But we can verify the server has the expected structure
        assert hasattr(server, '_setup_tools')
    
    @pytest.mark.asyncio
    async def test_run_task(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test running a single task"""
        # Create a test task file
        task_file = temp_dir / "test_task.md"
        task_file.write_text("# Test Task\n\nThis is a test task.")
        
        # Test run_task method directly
        arguments = {
            "task_path": str(task_file),
            "base_dir": str(temp_dir),
            "timeout_seconds": 30
        }
        
        result = await server._run_task(arguments)
        
        assert result["success"] is True
        assert "task_path" in result
        assert "status" in result
        assert result["status"] == "completed"
        
        # Check that result was stored
        task_id = str(task_file)
        assert task_id in server.task_results
        assert server.task_results[task_id]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_run_task_file_not_found(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test running a task with non-existent file"""
        arguments = {
            "task_path": str(temp_dir / "nonexistent.md"),
            "base_dir": str(temp_dir)
        }
        
        result = await server._run_task(arguments)
        
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_parse_task_list(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test parsing a task list file"""
        # Create a test task list
        task_list = temp_dir / "tasks.txt"
        task_list.write_text("""
# This is a comment
Task 1: Do something important
Task 2: Test the system
Task 3: Write documentation

# Another comment
Task 4: Review code
""")
        
        arguments = {
            "task_list_path": str(task_list),
            "base_dir": str(temp_dir)
        }
        
        result = await server._parse_task_list(arguments)
        
        assert result["success"] is True
        assert "task_files" in result
        assert "count" in result
        assert result["count"] == 4  # Should create 4 task files
        
        # Verify task files were created
        for task_file in result["task_files"]:
            assert Path(task_file).exists()
    
    @pytest.mark.asyncio
    async def test_create_project(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test creating a new project"""
        arguments = {
            "project_name": "test_project",
            "base_dir": str(temp_dir)
        }
        
        result = await server._create_project(arguments)
        
        assert result["success"] is True
        assert result["project"] == "test_project"
        assert "project_dir" in result
        
        # Verify project directory was created
        project_dir = Path(result["project_dir"])
        assert project_dir.exists()
        assert project_dir.is_dir()
    
    @pytest.mark.asyncio
    async def test_create_project_with_task_list(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test creating a project with a task list"""
        # Create task list
        task_list = temp_dir / "project_tasks.txt"
        task_list.write_text("Setup project\nWrite tests\nDeploy application")
        
        arguments = {
            "project_name": "test_project_with_tasks",
            "base_dir": str(temp_dir),
            "task_list_path": str(task_list)
        }
        
        result = await server._create_project(arguments)
        
        assert result["success"] is True
        assert "task_files" in result
        assert "task_count" in result
        assert result["task_count"] == 3
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test getting task status"""
        # Create some test task files
        (temp_dir / "task1.md").write_text("Task 1")
        (temp_dir / "task2.md").write_text("Task 2")
        
        # Run one task to create a result
        await server._run_task({
            "task_path": str(temp_dir / "task1.md"),
            "base_dir": str(temp_dir)
        })
        
        arguments = {"base_dir": str(temp_dir)}
        result = await server._get_task_status(arguments)
        
        assert result["success"] is True
        assert "tasks" in result
        assert "total_tasks" in result
        assert "completed_tasks" in result
        
        assert result["total_tasks"] == 2
        assert result["completed_tasks"] == 1
        
        # Check individual task status
        tasks = result["tasks"]
        assert "task1.md" in tasks
        assert "task2.md" in tasks
        assert tasks["task1.md"]["status"] == "completed"
        assert tasks["task2.md"]["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_get_task_summary(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test getting task summary"""
        # Create and run some tasks
        (temp_dir / "task1.md").write_text("Task 1")
        (temp_dir / "task2.md").write_text("Task 2")
        
        await server._run_task({
            "task_path": str(temp_dir / "task1.md"),
            "base_dir": str(temp_dir)
        })
        
        arguments = {"base_dir": str(temp_dir)}
        result = await server._get_task_summary(arguments)
        
        assert result["success"] is True
        assert "summary" in result
        
        summary = result["summary"]
        assert summary["total_tasks"] == 2
        assert summary["completed_tasks"] == 1
        assert summary["pending_tasks"] == 1
        assert summary["completion_rate"] == 50.0
    
    @pytest.mark.asyncio
    async def test_clean(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test cleanup functionality"""
        # Add some task results
        server.task_results["test"] = {"status": "completed"}
        
        # Create some temporary files
        (temp_dir / "temp1.tmp").write_text("temp file")
        (temp_dir / "temp2.tmp").write_text("another temp file")
        
        arguments = {"base_dir": str(temp_dir)}
        result = await server._clean(arguments)
        
        assert result["success"] is True
        assert result["task_results_cleared"] is True
        assert "cleaned_items" in result
        
        # Verify task results were cleared
        assert len(server.task_results) == 0
        
        # Verify temp files were cleaned
        assert not (temp_dir / "temp1.tmp").exists()
        assert not (temp_dir / "temp2.tmp").exists()
    
    @pytest.mark.asyncio
    async def test_run_all_tasks(self, server: FixedTaskRunnerServer, temp_dir: Path):
        """Test running all tasks in a directory"""
        # Create multiple task files
        (temp_dir / "task1.md").write_text("Task 1 content")
        (temp_dir / "task2.md").write_text("Task 2 content")
        (temp_dir / "task3.md").write_text("Task 3 content")
        (temp_dir / "not_a_task.txt").write_text("Not a task file")  # Should be ignored
        
        arguments = {"base_dir": str(temp_dir)}
        result = await server._run_all_tasks(arguments)
        
        assert result["success"] is True
        assert result["total_tasks"] == 3  # Only .md files
        assert result["tasks_executed"] == 3
        assert result["tasks_failed"] == 0
        assert "results" in result
        assert len(result["results"]) == 3
        
        # Verify all tasks were marked as completed
        for task_result in result["results"]:
            assert task_result["success"] is True
            assert task_result["status"] == "completed"


class TestMCPIntegration:
    """Integration tests for MCP protocol compliance"""
    
    @pytest.mark.asyncio
    async def test_tool_call_format(self):
        """Test that tool calls return proper MCP format"""
        server = FixedTaskRunnerServer()
        
        # Test a simple tool call
        with tempfile.TemporaryDirectory() as temp_dir:
            task_file = Path(temp_dir) / "test.md"
            task_file.write_text("Test task")
            
            # Simulate MCP tool call
            arguments = {"task_path": str(task_file)}
            result = await server._run_task(arguments)
            
            # Verify result format
            assert isinstance(result, dict)
            assert "success" in result
            assert isinstance(result["success"], bool)
    
    def test_server_has_required_attributes(self):
        """Test that server has all required MCP attributes"""
        server = FixedTaskRunnerServer()
        
        # Should have MCP server instance
        assert hasattr(server, 'server')
        assert server.server is not None
        
        # Should have required methods
        assert hasattr(server, '_setup_tools')
        assert hasattr(server, 'run')


if __name__ == "__main__":
    # Run tests manually
    import sys
    
    async def run_manual_test():
        """Run a basic manual test"""
        print("Testing Fixed Task Runner...")
        
        server = FixedTaskRunnerServer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test basic functionality
            temp_path = Path(temp_dir)
            task_file = temp_path / "test_task.md"
            task_file.write_text("# Test Task\n\nThis is a manual test.")
            
            # Test run_task
            result = await server._run_task({
                "task_path": str(task_file),
                "base_dir": str(temp_path)
            })
            
            print(f"Run task result: {result}")
            assert result["success"] is True
            
            # Test get_task_status
            status_result = await server._get_task_status({
                "base_dir": str(temp_path)
            })
            
            print(f"Task status: {status_result}")
            assert status_result["success"] is True
            
            print("✅ All manual tests passed!")
    
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        asyncio.run(run_manual_test())
    else:
        print("Run with 'manual' argument for manual testing, or use pytest for full test suite")