#!/usr/bin/env python3
"""
SutazAI System Setup Configuration
Supports optional features through extras_require
"""

from setuptools import setup, find_packages
import os

# Read the main requirements
def read_requirements(filename):
    """Read requirements from a file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

# Base requirements (always installed)
install_requires = read_requirements('requirements.txt')

# Optional requirements (installed with pip install .[feature])
extras_require = {
    # Core optional features
    'fsdp': [
        'fms-fsdp==0.1.5',
        'fairscale==0.4.13',
        'torch-distributed==2.5.1',
    ],
    'tabby': [
        'tabby-client==0.3.0',
        'httpx==0.27.2',
        'aiofiles==24.1.0',
    ],
    'gpu': [
        'torch==2.5.1+cu118',
        'nvidia-ml-py==12.560.30',
        'gpustat==1.1.1',
        'pynvml==11.5.3',
    ],
    'monitoring': [
        'opentelemetry-api==1.28.2',
        'opentelemetry-sdk==1.28.2',
        'opentelemetry-exporter-prometheus==0.49b2',
        'opentelemetry-instrumentation-fastapi==0.49b2',
    ],
    'enterprise': [
        'ldap3==2.9.1',
        'python-saml==1.16.0',
        'oauthlib==3.2.2',
        'authlib==1.3.2',
    ],
    'cloud': [
        'boto3==1.35.78',
        'azure-storage-blob==12.24.0',
        'google-cloud-storage==2.19.0',
        'oci==2.138.2',
    ],
    'notifications': [
        'slack-sdk==3.34.0',
        'discord.py==2.4.0',
        'python-telegram-bot==21.7',
        'twilio==9.3.7',
    ],
    'dev': read_requirements('requirements-dev.txt'),
    'test': read_requirements('requirements-test.txt'),
}

# All optional features
extras_require['all'] = list(set(sum(extras_require.values(), [])))

# Minimal installation (no optional features)
extras_require['minimal'] = []

setup(
    name='sutazai',
    version='60.0.0',
    description='SutazAI - Autonomous AI System with Optional Features',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='SutazAI Team',
    author_email='team@sutazai.com',
    url='https://github.com/sutazai/sutazai',
    packages=find_packages(exclude=['tests*', 'docs*']),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points={
        'console_scripts': [
            'sutazai=backend.app.main:main',
            'sutazai-backend=backend.app.main:main',
            'sutazai-agent=agents.core.base_agent:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# Installation instructions:
# 
# Minimal installation (no optional features):
#   pip install .
#
# With specific features:
#   pip install .[fsdp]           # FSDP distributed training
#   pip install .[tabby]          # TabbyML code completion
#   pip install .[gpu]            # GPU support
#   pip install .[monitoring]     # Advanced monitoring
#
# Multiple features:
#   pip install .[fsdp,tabby]     # FSDP and TabbyML
#
# All optional features:
#   pip install .[all]
#
# Development installation:
#   pip install -e .[dev,test]