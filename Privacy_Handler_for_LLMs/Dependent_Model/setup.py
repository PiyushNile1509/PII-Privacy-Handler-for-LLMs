#!/usr/bin/env python3
"""
Setup script for PII Privacy Handler
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pii-privacy-handler",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent PII Privacy Handler with Context-Aware Masking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pii-privacy-handler",
    packages=find_packages(),
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
        "Topic :: Security",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "streamlit": [
            "streamlit>=1.25.0",
        ],
        "gpu": [
            "tensorflow-gpu==2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pii-privacy-handler=src.privacy_handler:main",
            "pii-train=train_model:main",
            "pii-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
)