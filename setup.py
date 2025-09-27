"""
Setup script for Theory of Mind Benchmark for LLMs
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="theory-of-mind-benchmark",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive Theory of Mind evaluation suite for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/theory-of-mind-benchmark",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pre-commit>=2.20.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tom-benchmark=examples.run_benchmark:main",
        ],
    },
    keywords="theory-of-mind, ai-safety, clinical-psychology, llm-evaluation, social-cognition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/theory-of-mind-benchmark/issues",
        "Source": "https://github.com/yourusername/theory-of-mind-benchmark",
        "Documentation": "https://github.com/yourusername/theory-of-mind-benchmark#readme",
    },
    include_package_data=True,
    package_data={
        "": ["data/*.csv"],
    },
)