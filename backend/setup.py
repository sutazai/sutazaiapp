from setuptools import setup, find_packages

setup(
    name="sutazai-backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.9",
        "pydantic>=2.10.6",
        "uvicorn>=0.34.0",
        "loguru>=0.7.3",
        "starlette>=0.45.3",
        "typing-extensions>=4.12.2",
        "annotated-types>=0.7.0",
        "pydantic-core>=2.27.2",
        "click>=8.1.8",
        "h11>=0.14.0",
        "anyio>=4.8.0",
        "idna>=3.10",
        "sniffio>=1.3.1",
    ],
)
