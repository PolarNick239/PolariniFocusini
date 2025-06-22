from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import requests
from tqdm.auto import tqdm
from platformdirs import user_cache_dir   # ← cross-platform cache folder


# ─────────────────────── depth-estimation utilities ──────────────────────────
@dataclass
class DepthEstimationConfig:
    # App metadata for platformdirs
    app_name  : str = "PolariniFocusini"
    app_author: str = "polarnick"

    # Remote weights
    model_url: str = (
        "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model_q4f16.onnx"
    )

    # ONNX Runtime options
    input_size: tuple[int, int] = (518, 518)  # (W, H)
    provider  : str = "CPUExecutionProvider" # TODO support "CUDAExecutionProvider" (if available)

    # ───────────── derived helpers (don’t edit) ────────────────────────────
    @property
    def local_model_path(self) -> Path:
        """
        Per-user, per-OS cache path, e.g.

        • Linux  : ~/.cache/depth-anything/model_q4f16.onnx
        • Windows: %LOCALAPPDATA%/depth-anything/model_q4f16.onnx
        • macOS  : ~/Library/Caches/depth-anything/model_q4f16.onnx
        """
        cache_root = Path(user_cache_dir(self.app_name, self.app_author))
        return cache_root / Path(self.model_url).name


# ───────────────────────────── helpers ────────────────────────────────────────
def _ensure_weights(cfg: DepthEstimationConfig) -> str:
    """
    Make sure the ONNX file exists locally; download once if missing.

    Returns
    -------
    str : absolute path to the ONNX file
    """
    dst = cfg.local_model_path
    if dst.exists():                      # already cached → reuse
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading depth-anything weights to {dst} …")

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with requests.get(cfg.model_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar   = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                     desc=dst.name, leave=True)

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()

    tmp.rename(dst)
    return str(dst)


# ─────────────────────────── inference API ────────────────────────────────────
def _estimate_depth(
    bgr: np.ndarray,
    cfg: DepthEstimationConfig = DepthEstimationConfig(),
) -> np.ndarray:
    """
    Run the ONNX depth-estimation model and return a float32 depth/disparity map
    in the *original* image resolution.

    Parameters
    ----------
    bgr : uint8 H×W×3 image in OpenCV BGR order
    cfg : DepthEstimationConfig (path, input-size, provider)

    Returns
    -------
    depth : float32 H×W  (same height/width as *bgr*)
    """
    # 0) prepare ONNX Runtime session (cached after the first call)
    if not hasattr(_estimate_depth, "_sess"):
        model_path = _ensure_weights(cfg)
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        _estimate_depth._sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=[cfg.provider]
        )
        _estimate_depth._inp  = _estimate_depth._sess.get_inputs()[0].name
        _estimate_depth._out  = _estimate_depth._sess.get_outputs()[0].name

    # 1) preprocess
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, cfg.input_size, interpolation=cv2.INTER_AREA)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    x   = np.transpose(rgb, (2, 0, 1))[None]  # NCHW float32

    # 2) inference
    disp = _estimate_depth._sess.run(
        [_estimate_depth._out], {_estimate_depth._inp: x}
    )[0][0]                                     # (h,w) in model input size

    # 3) resize back to the original resolution & return
    h, w = bgr.shape[:2]
    disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return disp