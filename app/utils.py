def add(x, y):
    """Adds two numbers."""
    return x + y


def divide(x, y):
    """Divides two numbers, handling ZeroDivisionError."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y
