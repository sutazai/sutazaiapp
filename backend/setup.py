"""
Setup configuration for SutazAI backend with optional dependencies
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="sutazai-backend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=required,
    extras_require={
        "fsdp": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.0.0",
            "fairscale>=0.4.0",
        ],
        "tabby": [
            # httpx already in main requirements
        ],
        "ml": [
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
        "all": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.0.0",
            "fairscale>=0.4.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ]
    },
    python_requires=">=3.10",
    description="SutazAI Backend with optional ML features",
    author="SutazAI Team",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)