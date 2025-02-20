import functools
import inspect
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import Protocol, runtime_checkable

T = TypeVar("T")
R = TypeVar("R")
V = TypeVar("V")


@runtime_checkable
class TypeValidator(Protocol):
    """Protocol for type validation strategies."""

    def __call__(self, value: Any) -> bool:
        """
        Validate a value.

        Args:
            value (Any): Value to validate

        Returns:
            bool: Whether the value is valid
        """
        return True


class SafeOptional(Generic[T]):
    """
    Advanced optional type wrapper with safe access and default value handling.

    Provides:
    - Safe value extraction
    - Default value management
    - Type preservation
    - Comprehensive validation
    """

    def __init__(self, default: Optional[T] = None):
        self._default = default

    def __call__(self, value: Optional[T]) -> T:
        """
        Safely extract value with optional default fallback.

        Args:
            value (Optional[T]): Input value to validate

        Returns:
            T: Validated value or default

        Raises:
            ValueError: If value is None and no default is set
        """
        if value is None:
            if self._default is None:
                raise ValueError("No value provided and no default set")
            return cast(T, self._default)
        return value


def safe_optional(default: T) -> SafeOptional[T]:
    """
    Create a safe optional type handler with a default value.

    Args:
        default (T): Default value to use if no value is provided

    Returns:
        SafeOptional[T]: Optional type handler
    """
    return SafeOptional(default)


def type_validated(
    validators: Optional[Dict[str, TypeValidator]] = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator for comprehensive type validation of function arguments and return values.

    Args:
        validators (Optional[Dict[str, TypeValidator]]): Custom type validators

    Returns:
        Callable: Decorated function with type validation
    """
    validators = validators or {}

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Validate input arguments
            sig = inspect.signature(func)
            bound_arguments = sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            for param_name, param_value in bound_arguments.arguments.items():
                if param_name in validators:
                    validator = validators[param_name]
                    if not validator(param_value):
                        raise TypeError(
                            f"Invalid type for {param_name}: {type(param_value)}"
                        )

            # Execute function
            result = func(*args, **kwargs)

            # Optional return type validation
            return_validator = validators.get("return")
            if return_validator and not return_validator(result):
                raise TypeError(f"Invalid return type: {type(result)}")

            return result

        return wrapper

    return decorator


def safe_type_conversion(
    value: Any,
    target_type: Type[T],
    default: Optional[T] = None,
    strict: bool = False,
) -> Optional[T]:
    """
    Safely convert value to target type with comprehensive error handling.

    Args:
        value (Any): Value to convert
        target_type (Type[T]): Target type for conversion
        default (Optional[T]): Default value if conversion fails
        strict (bool): Whether to raise exceptions or return default

    Returns:
        Optional[T]: Converted value or default

    Raises:
        TypeError: If conversion fails and strict mode is enabled
    """
    try:
        if isinstance(value, target_type):
            return value

        # Special handling for generic types
        origin = get_origin(target_type)
        if origin is not None:
            args = get_args(target_type)
            if origin is Union and type(None) in args:
                # Handle Optional types
                non_none_type = next(t for t in args if t is not type(None))
                return safe_type_conversion(
                    value, non_none_type, default, strict
                )

        # Handle callable types
        if callable(target_type):
            try:
                return target_type(value)  # type: ignore
            except Exception as e:
                if strict:
                    raise TypeError(
                        f"Cannot convert {value} to {target_type}"
                    ) from e
                return default

        return value if isinstance(value, target_type) else target_type(value)

    except (TypeError, ValueError) as e:
        if strict:
            raise TypeError(f"Cannot convert {value} to {target_type}") from e

        logging.warning(f"Type conversion failed: {e}")
        return default


def validate_nested_structure(
    data: Any,
    expected_structure: Dict[str, Type[Any]],
    allow_extra_keys: bool = False,
) -> bool:
    """
    Validate complex nested data structures with comprehensive type checking.

    Args:
        data (Any): Data to validate
        expected_structure (Dict[str, Type[Any]]): Expected structure and types
        allow_extra_keys (bool): Whether additional keys are permitted

    Returns:
        bool: Whether data matches expected structure
    """
    if not isinstance(data, dict):
        return False

    for key, expected_type in expected_structure.items():
        if key not in data:
            return False

        value = data[key]
        if value is None and get_origin(expected_type) is Union:
            # Handle Optional types
            continue

        if not isinstance(value, expected_type):
            # Try type conversion for basic types
            try:
                converted = safe_type_conversion(value, expected_type)
                if converted is None:
                    return False
            except (TypeError, ValueError):
                return False

    if not allow_extra_keys:
        if set(data.keys()) != set(expected_structure.keys()):
            return False

    return True


# Comprehensive type checking utilities
__all__ = [
    "SafeOptional",
    "safe_optional",
    "type_validated",
    "safe_type_conversion",
    "validate_nested_structure",
]
