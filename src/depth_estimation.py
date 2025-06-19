import os
import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass


# ─────────────────────── depth-estimation utilities ──────────────────────────
@dataclass
class DepthEstimationConfig:
    model_path: str = os.path.join("models", "model_q4f16.onnx")
    input_size: tuple[int, int] = (518, 518)        # (W, H)
    provider: str = "CPUExecutionProvider"          # TODO support and use "CUDAExecutionProvider" (if available)

def _estimate_depth(bgr: np.ndarray,
                    cfg: DepthEstimationConfig = DepthEstimationConfig()
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
        opt = ort.SessionOptions();  opt.log_severity_level = 3
        _estimate_depth._sess = ort.InferenceSession(
            cfg.model_path, sess_options=opt, providers=[cfg.provider])
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