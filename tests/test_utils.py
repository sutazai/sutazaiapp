import pytest

from app.utils import add, divide


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_divide():
    assert divide(10, 2) == 5.0
    with pytest.raises(ValueError):
        divide(1, 0)

# Example of a negative test
def test_add_with_string():
    with pytest.raises(TypeError):
        add("hello", 5) 