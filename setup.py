import re
from pathlib import Path

from setuptools import find_packages, setup

def find_version():
    init_py = Path(__file__).parent / "src" / "polarini_focusini" / "__init__.py"
    match = re.search(r'^__version__ = ["\']([^"\']+)["\']', init_py.read_text(), re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Version string not found!")

# ------------------------------------------------------------------------------
NAME        = "polarini-focusini"          #  ↖ what users will 'pip install'
VERSION     = find_version()
DESCRIPTION = (
    "Detect in-focus regions in images using depth estimation "
    "and Difference-of-Gaussian (DoG) extrema voting across depth bins."
)
HERE        = Path(__file__).parent
LONG_DESC   = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Nikolai Poliarnyi",
    license="MIT",
    python_requires=">=3.9",
    url="https://github.com/PolarNick239/PolariniFocusini",
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # ────────────────────────── dependencies ────────────────────────── #
    install_requires=[
        "onnxruntime>=1.17.0",       # **mandatory** CPU runtime
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "platformdirs>=4.2.0",
    ],
    extras_require={
        # Users with GPUs can install:  pip install polarini-focusini[cuda]
        "cuda": [
            "onnxruntime-gpu>=1.17.0",
            "onnxruntime-cuda>=1.17.0",
        ],
        "dev": [
            "pytest>=8.0.0",         # run tests
            "packaging>=24.0",       # version check helper used in tests
        ],
    },

    # ────────────────────────── CLI entry point ─────────────────────── #
    entry_points={
        "console_scripts": [
            "polarini-focusini = polarini_focusini._cli:main",
        ],
    },

    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
