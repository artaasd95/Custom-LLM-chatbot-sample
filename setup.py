#!/usr/bin/env python3
"""Setup script for Custom LLM Chatbot."""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Filter out platform-specific requirements for base installation
base_requirements = []
for req in requirements:
    if '; sys_platform' not in req:
        base_requirements.append(req)
    else:
        # Add the base requirement without platform restriction
        base_req = req.split(';')[0].strip()
        base_requirements.append(base_req)

# Remove duplicates while preserving order
seen = set()
filtered_requirements = []
for req in base_requirements:
    req_name = req.split('>=')[0].split('==')[0].split('[')[0]
    if req_name not in seen:
        seen.add(req_name)
        filtered_requirements.append(req)

setup(
    name="custom-llm-chatbot",
    version="1.0.0",
    author="Custom LLM Team",
    author_email="team@customllm.com",
    description="A comprehensive framework for training, fine-tuning, and serving Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artaasd95/Custom-LLM-chatbot-sample",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=filtered_requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "serving": [
            "vllm>=0.2.0; sys_platform != 'win32'",
            "onnxruntime-gpu>=1.15.0; sys_platform != 'win32'",
        ],
        "distributed": [
            "deepspeed>=0.9.0; sys_platform != 'win32'",
        ],
        "optimized": [
            "flash-attn>=2.0.0; sys_platform != 'win32'",
            "xformers>=0.0.20; sys_platform != 'win32'",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
        ],
        "all": [
            "vllm>=0.2.0; sys_platform != 'win32'",
            "onnxruntime-gpu>=1.15.0; sys_platform != 'win32'",
            "deepspeed>=0.9.0; sys_platform != 'win32'",
            "flash-attn>=2.0.0; sys_platform != 'win32'",
            "xformers>=0.0.20; sys_platform != 'win32'",
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=train:main",
            "llm-serve=serve:main",
            "llm-example=example_usage:main",
        ],
    },
    include_package_data=True,
    package_data={
        "custom_llm_chatbot": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "llm",
        "large language model",
        "chatbot",
        "training",
        "fine-tuning",
        "serving",
        "inference",
        "transformers",
        "pytorch",
        "machine learning",
        "artificial intelligence",
        "nlp",
        "natural language processing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/artaasd95/Custom-LLM-chatbot-sample/issues",
        "Source": "https://github.com/artaasd95/Custom-LLM-chatbot-sample",
        "Documentation": "https://custom-llm-chatbot.readthedocs.io/",
    },
)