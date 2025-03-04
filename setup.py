#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="sutazaiapp",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pytest-html",
    ],
    python_requires=">=3.11",
) 