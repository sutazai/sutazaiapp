from setuptools import setup, find_packages

setup(
    name="sutazai_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    extras_require={"dev": ["pylint", "mypy", "black", "isort", "bandit", "pytest", "ruff"]},
)
