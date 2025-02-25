import sys
import textwrap
from typing import List, Optional, Sequence

# Shim to wrap setup.py invocation with setuptools
# Note that __file__ is handled via two {!r} *and* %r, to ensure that paths on
# Windows are correctly handled (it should be "C:\\Users" not "C:\Users").
_SETUPTOOLS_SHIM = textwrap.dedent(
    """
    exec(compile('''
    # This is <pip-setuptools-caller> -- a caller that pip uses to run setup.py
    #
    # - It imports setuptools before invoking setup.py, to enable projects that directly
    #   import from `distutils.core` to work with newer packaging standards.
    # - It provides a clear error message when setuptools is not installed.
    # - It sets `sys.argv[0]` to the underlying `setup.py`, when invoking `setup.py` so
    #   setuptools doesn't think the script is `-c`. This avoids the following warning:
    #     manifest_maker: standard file '-c' not found".
    # - It generates a shim setup.py, for handling setup.cfg-only projects.
    import os, sys, tokenize

    try:
        import setuptools
    except ImportError as error:
        print(
            "ERROR: Can not execute `setup.py` since setuptools is not available in "
            "the build environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    __file__ = %r
    sys.argv[0] = __file__

    if os.path.exists(__file__):
        filename = __file__
        with tokenize.open(__file__) as f:
            setup_py_code = f.read()
    else:
        filename = "<auto-generated setuptools caller>"
        setup_py_code = "from setuptools import setup; setup()"

    exec(compile(setup_py_code, filename, "exec"))
    ''' % ({!r},), "<pip-setuptools-caller>", "exec"))
    """
).rstrip()


def make_setuptools_shim_args(
    setup_py_path: str,
    global_options: Optional[Sequence[str]] = None,
    no_user_config: bool = False,
    unbuffered_output: bool = False,
) -> List[str]:
    """
    Get setuptools command arguments with shim wrapped setup file invocation.

    :param setup_py_path: The path to setup.py to be wrapped.
    :param global_options: Additional global options.
    :param no_user_config: If True, disables personal user configuration.
    :param unbuffered_output: If True, adds the unbuffered switch to the
     argument list.
    """
    args = [sys.executable]
    if unbuffered_output:
        args += ["-u"]
    args += ["-c", _SETUPTOOLS_SHIM.format(setup_py_path)]
    if global_options:
        args += global_options
    if no_user_config:
        args += ["--no-user-cfg"]
    return args


def make_setuptools_bdist_wheel_args(
    setup_py_path: str,
    global_options: Sequence[str],
    build_options: Sequence[str],
    destination_dir: str,
) -> List[str]:
    # NOTE: Eventually, we'd want to also -S to the flags here, when we're
    # isolating. Currently, it breaks Python in virtualenvs, because it
    # relies on site.py to find parts of the standard library outside the
    # virtualenv.
    args = make_setuptools_shim_args(
        setup_py_path, global_options=global_options, unbuffered_output=True
    )
    args += ["bdist_wheel", "-d", destination_dir]
    args += build_options
    return args


def make_setuptools_clean_args(
    setup_py_path: str,
    global_options: Sequence[str],
) -> List[str]:
    args = make_setuptools_shim_args(
        setup_py_path, global_options=global_options, unbuffered_output=True
    )
    args += ["clean", "--all"]
    return args


def make_setuptools_develop_args(
    setup_py_path: str,
    *,
    global_options: Sequence[str],
    no_user_config: bool,
    prefix: Optional[str],
    home: Optional[str],
    use_user_site: bool,
) -> List[str]:
    assert not (use_user_site and prefix)

    args = make_setuptools_shim_args(
        setup_py_path,
        global_options=global_options,
        no_user_config=no_user_config,
    )

    args += ["develop", "--no-deps"]

    if prefix:
        args += ["--prefix", prefix]
    if home is not None:
        args += ["--install-dir", home]

    if use_user_site:
        args += ["--user", "--prefix="]

    return args


def make_setuptools_egg_info_args(
    setup_py_path: str,
    egg_info_dir: Optional[str],
    no_user_config: bool,
) -> List[str]:
    args = make_setuptools_shim_args(
        setup_py_path, no_user_config=no_user_config
    )

    args += ["egg_info"]

    if egg_info_dir:
        args += ["--egg-base", egg_info_dir]

    return args
