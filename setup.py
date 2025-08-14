#!/usr/bin/env python3
"""
Setup script for SpeedNet - 3D-Aware Vehicle Speed Estimation
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "SpeedNet: 3D-Aware Vehicle Speed Estimation from Video"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="speednet",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="3D-Aware Vehicle Speed Estimation from Video",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/speednet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black",
            "flake8", 
            "pytest",
            "jupyter",
            "jupyterlab",
        ],
        "vis": [
            "plotly",
            "seaborn",
            "wandb",
        ],
        "web": [
            "fastapi",
            "uvicorn",
            "streamlit",
        ]
    },
    entry_points={
        "console_scripts": [
            "speednet-train=train_speednet:main",
            "speednet-infer=inference_pipeline:main",
            "speednet-analyze=dataset_analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "speednet": [
            "models/*.py",
            "configs/*.yaml",
            "assets/*",
        ],
    },
    zip_safe=False,
)