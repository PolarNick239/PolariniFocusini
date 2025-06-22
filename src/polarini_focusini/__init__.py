"""
Top-level package for polarini-focusini.

Usage
-----
>>> from polarini_focusini import detect_infocus_mask
"""
from .infocus_detection import detect_infocus_mask  # re-export  :contentReference[oaicite:1]{index=1}

__all__ = ["detect_infocus_mask"]
__version__ = "0.1.1"