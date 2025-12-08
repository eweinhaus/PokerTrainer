#!/usr/bin/env python3
"""
Setup script for rlcard_modified package.
This package provides a modified version of RLCard with BB-first action order support.
"""

from setuptools import setup, find_packages

setup(
    name="rlcard_modified",
    version="1.0.0",
    description="Modified RLCard with BB-first action order support",
    author="AI Poker Coach Team",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "rlcard>=1.0.0",
    ],
)

