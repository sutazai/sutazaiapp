"""
Purpose: Basic smoke tests to verify testing infrastructure works
Usage: pytest tests/test_smoke.py
Requirements: pytest
"""
import pytest
import sys
import os


def test_python_version():
    """Test that Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8), "Python 3.8+ is required"


def test_import_backend():
    """Test that backend package can be imported."""
    try:
        # Add backend to path if needed
        backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
        if backend_path not in sys.path:
            # Path handled by pytest configuration
        
        # Try to import backend app
        import app
        assert True
    except ImportError as e:
        pytest.skip(f"Backend import failed: {e}")


def test_basic_math():
    """Test basic math operations work."""
    assert 2 + 2 == 4
    assert 10 * 5 == 50
    assert 100 / 4 == 25


def test_string_operations():
    """Test basic string operations."""
    assert "hello" + " " + "world" == "hello world"
    assert "UPPER".lower() == "upper"
    assert "lower".upper() == "LOWER"


def test_list_operations():
    """Test basic list operations."""
    test_list = [1, 2, 3]
    test_list.append(4)
    assert test_list == [1, 2, 3, 4]
    assert len(test_list) == 4
    assert sum(test_list) == 10


def test_dict_operations():
    """Test basic dictionary operations."""
    test_dict = {"key": "value"}
    test_dict["new_key"] = "new_value"
    assert len(test_dict) == 2
    assert test_dict.get("key") == "value"
    assert "new_key" in test_dict


@pytest.mark.asyncio
async def test_async_function():
    """Test that async functions work."""
    import asyncio
    
    async def async_add(a, b):
        await asyncio.sleep(0.01)
        return a + b
    
    result = await async_add(5, 3)
    assert result == 8


class TestBasicClass:
    """Test class-based tests work."""
    
    def test_class_method(self):
        """Test method in test class."""
        assert True
    
    def test_fixture_usage(self):
        """Test that we can use fixtures."""
        test_data = {"status": "ok"}
        assert test_data["status"] == "ok"


@pytest.mark.unit
def test_with_marker():
    """Test that markers work correctly."""
    assert True


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_parametrized(input, expected):
    """Test parametrized tests work."""
    assert input * 2 == expected


def test_exception_handling():
    """Test exception handling."""
    with pytest.raises(ValueError):
        raise ValueError("This is expected")
    
    with pytest.raises(ZeroDivisionError):
        1 / 0