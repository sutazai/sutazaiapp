from distutils.core import setup

setup(
    name="sutazai_app",
    version="0.1.0",
    packages=[
        "core_system",
        "core_system.monitoring",
        "core_system.utils",
        "core_system.dependency_management",
        "misc",
        "misc.config",
        "system_integration",
    ],
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
