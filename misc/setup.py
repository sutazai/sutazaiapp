from setuptools import find_packages, setup

setup(
    name="sutazai_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langdetect>=1.0.9",
        "pyperclip>=1.8.2",
    ],
    python_requires=">=3.11",
    extras_require={
        "dev": [
            "pylint>=3.0.0",
            "black>=25.1.0",
            "mypy>=1.8.0",
        ],
    },
)
