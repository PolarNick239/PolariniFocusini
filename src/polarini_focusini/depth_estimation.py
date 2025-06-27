import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import requests
from tqdm.auto import tqdm
from platformdirs import user_cache_dir   # ← cross-platform cache folder

# ──────────────────────────── presets / enums ────────────────────────────────
class DepthModel(str, Enum):
    """Enumeration of the supported monocular depth back‑ends."""

    DEPTH_ANYTHING = "depth_anything"
    DEPTH_PRO = "depth_pro"


# (url, (W,H), mean, std)
_MODEL_PRESETS: Dict[DepthModel, Tuple[str, Tuple[int, int], np.ndarray, np.ndarray]] = {
    DepthModel.DEPTH_ANYTHING: (
        "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model_q4f16.onnx",
        (518, 518),
        np.array([0.485, 0.456, 0.406], dtype=np.float32),
        np.array([0.229, 0.224, 0.225], dtype=np.float32),
    ),
    DepthModel.DEPTH_PRO: (
        "https://huggingface.co/onnx-community/DepthPro-ONNX/resolve/main/onnx/model_q4f16.onnx",
        (1536, 1536),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
    ),
}


# ─────────────────────── depth‑estimation utilities ─────────────────────────
@dataclass
class DepthEstimationConfig:
    """Configuration for the depth estimator.

    Unspecified fields are filled from the *model* preset so the default
    behaviour stays the same even after adding new back‑ends.
    """

    model: DepthModel = DepthModel.DEPTH_ANYTHING

    # fine‑grained options (may be overridden manually)
    model_url: Optional[str] = None
    input_size: Optional[Tuple[int, int]] = None  # (W, H)
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    # cache organisation
    app_name: str = "polarini_focusini"
    app_author: str = "polarnick"

    def __post_init__(self) -> None:
        # Populate defaults from preset
        url, size, mean, std = _MODEL_PRESETS[self.model]
        if self.model_url is None:
            self.model_url = url
        if self.input_size is None:
            self.input_size = size
        if self.mean is None:
            self.mean = mean
        if self.std is None:
            self.std = std

    # ───────────────────── derived helpers ────────────────────────────────
    @property
    def local_model_path(self) -> Path:
        """
        Per-user, per-OS cache path, e.g.

        • Linux  : ~/.cache/.../model_q4f16.onnx
        • Windows: %LOCALAPPDATA%/.../model_q4f16.onnx
        • macOS  : ~/Library/Caches/.../model_q4f16.onnx
        """
        cache_root = Path(user_cache_dir(self.app_name, self.app_author))

        # Note that we want to add model name (self.model.value - i.e. depth_anything or depth_pro),
        # so that no ONNX model weights collision happens
        return cache_root / self.model.value / Path(self.model_url).name

# ───────────────────────────── helpers ────────────────────────────────────────
def _ensure_weights(cfg: DepthEstimationConfig) -> str:
    """
    Make sure the ONNX file exists locally; download once if missing.

    Returns
    -------
    str : absolute path to the ONNX file
    """
    dst = cfg.local_model_path
    if dst.exists():
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {cfg.model.value} weights to {dst} …")

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with requests.get(cfg.model_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dst.name)
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
        bar.close()

    tmp.rename(dst)
    return str(dst)


# ─────────────────────────── inference API ───────────────────────────────────

def _estimate_depth(bgr: np.ndarray, cfg: Optional[DepthEstimationConfig] = None, *,
                    ignore_cuda = False, verbose = False) -> np.ndarray:
    """Estimate depth/disparity map for *bgr* using the chosen backend."""

    if cfg is None:
        cfg = DepthEstimationConfig()  # default = Depth‑Anything

    # 0) obtain/cached ONNX session
    if not hasattr(_estimate_depth, "_cache"):
        _estimate_depth._cache = {}

    model_path = _ensure_weights(cfg)
    if model_path not in _estimate_depth._cache:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # quiet

        sess = None

        if not ignore_cuda:
            cuda_provider = "CUDAExecutionProvider"
            try:
                sess = ort.InferenceSession(model_path, sess_options=opts, providers=[cuda_provider])
                if verbose:
                    print("ONNX session providers:")
                    print(sess.get_providers())
                    print("ONNX: CUDA execution session attempted to be created")
            except RuntimeError as e:
                if "install" in e.args[0] and "CUDA" in e.args[0] and "cuDNN" in e.args[0]:
                    print("CUDA 12.2 can be downloaded here: https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local", file=sys.stderr)
                    print("cuDNN 9 can be downloaded here: https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip", file=sys.stderr)
                    print("unzip cuDNN 9 archive and copy all dlls from its bin subdir into C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin", file=sys.stderr)
                    print("You can read more about ONNX CUDA requirements here: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements", file=sys.stderr)
                    print("You can use --ignore-cuda to suppress this error", file=sys.stderr)
                    print("Can't use CUDA, falling back to CPU...", file=sys.stderr)

        if sess is None:
            cpu_provider = "CPUExecutionProvider"
            sess = ort.InferenceSession(model_path, sess_options=opts, providers=[cpu_provider])
            if verbose:
                print("ONNX: CPU execution session created")

        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        _estimate_depth._cache[model_path] = (sess, inp_name, out_name)

    sess, inp_name, out_name = _estimate_depth._cache[model_path]

    # 1) pre‑processing
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, cfg.input_size, interpolation=cv2.INTER_AREA)
    rgb = (rgb - cfg.mean) / cfg.std
    tensor = np.transpose(rgb, (2, 0, 1))[None]  # NCHW

    # 2) inference
    disp = sess.run([out_name], {inp_name: tensor})[0][0]  # (H, W)

    # 3) resize back to original resolution
    h, w = bgr.shape[:2]
    disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return disp
