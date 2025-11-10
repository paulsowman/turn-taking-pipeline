"""Setup script for turn-taking-pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="turn-taking-pipeline",
    version="0.1.0",
    description="End-to-end pipeline for analyzing turn-taking in conversational MEG data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Natural Conversations Study Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "mne>=1.5.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "praat-parselmouth>=0.4.3",
        "openai-whisper>=20230314",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "llm": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
        "modeling": [
            "bambi>=0.12.0",
            "pymc>=5.6.0",
            "lifelines>=0.27.0",
            "arviz>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-pipeline=scripts.run_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
