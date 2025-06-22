"""
Basic smoke tests for **infocus‑detect**.

Run with::

    pytest -q
"""

import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from polarini_focusini import detect_infocus_mask, __version__

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover
    Version = None


def test_version_pep440():
    """__version__ string follows PEP 440."""
    if Version is None:
        pytest.skip("packaging not installed")
    Version(__version__)


def test_detect_infocus_mask_smoke():
    """detect_infocus_mask returns a boolean mask of matching H×W."""
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    mask = detect_infocus_mask(img)
    assert mask.shape[:2] == img.shape[:2]
    assert mask.dtype == np.bool_


def test_cli_help_runs():
    """`python -m polarini_focusini._cli --help` exits with 0."""
    result = subprocess.run(
        [sys.executable, "-m", "polarini_focusini._cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Generate in-focus masks" in result.stdout
