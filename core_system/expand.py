"""
SutazAI Configuration Expansion Module
-------------------------------------
A simplified version of configuration expansion utilities for the SutazAI system.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable, Mapping
from configparser import ConfigParser
from glob import iglob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Type variables for generic functions
_K = TypeVar("_K")
_V = TypeVar("_V")
_R = TypeVar("_R")  # Return type for callables

# Type aliases
StrPath = Union[str, Path]
AnyCallable = Callable[..., Any]  # Generic callable type


def glob_relative(
    patterns: Iterable[str], root_dir: Optional[StrPath] = None
) -> List[str]:
    """
    Expand the list of glob patterns, but preserving relative paths.
    
    Args:
        patterns: List of glob patterns
        root_dir: Path to which globs should be relative (current directory by default)
        
    Returns:
        List of expanded paths
    """
    glob_characters = {"*", "?", "[", "]", "{", "}"}
    expanded_values = []
    root_dir = root_dir or os.getcwd()
    
    for value in patterns:
        # Has globby characters?
        if any(char in value for char in glob_characters):
            # then expand the glob pattern while keeping paths *relative*:
            glob_path = os.path.abspath(os.path.join(str(root_dir), value))
            expanded_values.extend(
                sorted(
                    os.path.relpath(path, str(root_dir)).replace(os.sep, "/")
                    for path in iglob(glob_path, recursive=True)
                )
            )
        else:
            # take the value as-is
            path = os.path.relpath(value, str(root_dir)).replace(os.sep, "/")
            expanded_values.append(path)
    
    return expanded_values


def read_files(
    filepaths: Union[StrPath, Iterable[StrPath]], 
    root_dir: Optional[StrPath] = None
) -> str:
    """
    Return the content of the files concatenated using ``\n`` as str.
    
    Args:
        filepaths: Path or list of paths to read
        root_dir: Root directory (current directory by default)
        
    Returns:
        Concatenated file contents
    """
    root_dir = os.path.abspath(str(root_dir or os.getcwd()))
    
    # Convert single path to list
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    
    # Convert paths to absolute paths
    abs_paths = [os.path.join(root_dir, str(path)) for path in filepaths]
    
    # Filter existing files
    existing_files = []
    for path in abs_paths:
        if os.path.isfile(path):
            existing_files.append(path)
        else:
            print(f"Warning: File {path!r} cannot be found")
    
    # Read files
    contents = []
    for path in existing_files:
        # Check that the file is within root_dir
        if Path(root_dir) not in Path(os.path.abspath(path)).parents:
            raise ValueError(
                f"Cannot access {path!r} (or anything outside {root_dir!r})"
            )
        
        with open(path, encoding="utf-8") as f:
            contents.append(f.read())
    
    return "\n".join(contents)


def read_attr(
    attr_desc: str,
    package_dir: Optional[Mapping[str, str]] = None,
    root_dir: Optional[StrPath] = None
) -> Any:
    """
    Reads the value of an attribute from a module.
    
    Args:
        attr_desc: Dot-separated string describing how to reach the attribute
        package_dir: Mapping of package names to their location in disk
        root_dir: Path to directory containing all the packages
        
    Returns:
        The attribute value
    """
    root_dir = root_dir or os.getcwd()
    attrs_path = attr_desc.strip().split(".")
    attr_name = attrs_path.pop()
    module_name = ".".join(attrs_path)
    module_name = module_name or "__init__"
    
    # Find the module
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        # Fallback to direct file import
        try:
            # Try to find the module file
            if package_dir and module_name in package_dir:
                module_path = os.path.join(str(root_dir), package_dir[module_name])
                module_path = os.path.join(module_path, "__init__.py")
            else:
                module_path = os.path.join(
                    str(root_dir), 
                    module_name.replace(".", os.sep) + ".py"
                )
            
            # If it's a directory, look for __init__.py
            if os.path.isdir(module_path):
                module_path = os.path.join(module_path, "__init__.py")
            
            # If the file doesn't exist, try adding .py extension
            if not os.path.exists(module_path) and not module_path.endswith(".py"):
                module_path += ".py"
            
            # If the file still doesn't exist, raise an error
            if not os.path.exists(module_path):
                raise ImportError(f"Module {module_name} not found")
            
            # Load the module from file
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load module {module_name} from {module_path}"
                )
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"Could not find attribute {attr_name} in module {module_name}"
            ) from e


def resolve_class(
    qualified_class_name: str,
    package_dir: Optional[Mapping[str, str]] = None,
    root_dir: Optional[StrPath] = None
) -> AnyCallable:
    """
    Given a qualified class name, return the associated class object.
    
    Args:
        qualified_class_name: Fully qualified class name
        package_dir: Mapping of package names to their location in disk
        root_dir: Path to directory containing all the packages
        
    Returns:
        The class object
    """
    # Just use read_attr directly since we're looking for a class
    result = read_attr(qualified_class_name, package_dir, root_dir)
    if not callable(result):
        raise TypeError(f"{qualified_class_name} is not callable")
    return cast(AnyCallable, result)


def cmdclass(
    values: Dict[str, str],
    package_dir: Optional[Mapping[str, str]] = None,
    root_dir: Optional[StrPath] = None
) -> Dict[str, AnyCallable]:
    """
    Given a dictionary mapping command names to strings for qualified class
    names, apply resolve_class to the dict values.
    
    Args:
        values: Dictionary mapping command names to qualified class names
        package_dir: Mapping of package names to their location in disk
        root_dir: Path to directory containing all the packages
        
    Returns:
        Dictionary mapping command names to class objects
    """
    return {k: resolve_class(v, package_dir, root_dir) for k, v in values.items()}


def version(value: Union[Callable[[], Any], Iterable[Union[str, int]], str]) -> str:
    """
    When getting the version directly from an attribute,
    it should be normalised to string.
    
    Args:
        value: Version value or callable that returns the version
        
    Returns:
        Normalized version string
    """
    _value = value() if callable(value) else value
    
    if isinstance(_value, str):
        return _value
    if hasattr(_value, "__iter__"):
        return ".".join(map(str, _value))
    return f"{_value}"


def entry_points(
    text: str, text_source: str = "entry-points"
) -> Dict[str, Dict[str, str]]:
    """
    Given the contents of entry-points file, process it into a 2-level dictionary.
    
    Args:
        text: Contents of entry-points file
        text_source: Source of the text (for error reporting)
        
    Returns:
        Dictionary mapping entry-point groups to dictionaries mapping 
        entry-point names to references
    """
    parser = ConfigParser()
    # Use a different approach to make optionxform case-sensitive
    parser.optionxform = lambda optionstr: optionstr  # type: ignore
    parser.read_string(text, text_source)
    
    groups = {}
    for section in parser.sections():
        groups[section] = dict(parser.items(section))
    
    return groups
