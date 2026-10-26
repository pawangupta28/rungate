"""Rungate — Runtime validation middleware for AI agent tool-calls."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rungate",
    version="0.1.0",
    author="Pawan Gupta",
    description="Runtime validation middleware for AI agent tool-calls",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pawan-kumar/rungate",
    project_urls={
        "Bug Tracker": "https://github.com/pawan-kumar/rungate/issues",
        "Documentation": "https://github.com/pawan-kumar/rungate#readme",
        "Source Code": "https://github.com/pawan-kumar/rungate",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",
    install_requires=[],  # Zero required dependencies — stdlib only
    extras_require={
        "yaml": ["pyyaml>=6.0"],
        "schema": ["jsonschema>=4.0"],
        "server": ["fastapi>=0.100", "uvicorn>=0.20"],
        "all": ["pyyaml>=6.0", "jsonschema>=4.0", "fastapi>=0.100", "uvicorn>=0.20"],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "pyyaml>=6.0",
            "jsonschema>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rungate-server=rungate.api.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    keywords="ai agent security validation tool-call middleware guardrails runtime",
    license="BUSL-1.1",
)
