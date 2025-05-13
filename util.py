from ultralytics import YOLO
from onnxruntime.quantization import shape_inference, quantize_dynamic, QuantType
import cv2
import numpy as np

def load_onnx_model(model_path:str, res:tuple = (640, 640)) -> YOLO:
    """
    Load the ONNX model using ONNX Runtime and return a YOLO wrapper.
    :param model_path: Path to the ONNX model file.
    :param res: Resolution for export (width, height).
    :return: YOLO wrapper for the ONNX model.
    """
    try:
        model = YOLO(model_path)
        model.export(format="onnx", nms=False, imgsz=res, dynamic=False, device="cpu")
        onnx_name = model_path.rsplit(".", 1)[0] + ".onnx"
        onnx_model = YOLO(onnx_name, task="detect")
        print("ONNX model loaded/exported successfully.")
        return onnx_model
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None
    

def quant_onnx(model_path:str, quant_model_path:str) -> YOLO:
    """
    Quantize the ONNX model using ONNX Runtime.
    :param model_path: Path to the ONNX model file.
    :param quant_model_path: Path to save the quantized model.
    :return: YOLO wrapper for the quantized model.
    """
    pre_path = model_path.rsplit(".", 1)[0] + "_pre.onnx"
    try:
        shape_inference.quant_pre_process(model_path, pre_path)
    except Exception as e:
        print(f"Error during shape inference: {e}")
        return None
    try:
        quantize_dynamic(pre_path, quant_model_path, weight_type=QuantType.QUInt8)
        print("Model quantization successful.")
        return YOLO(quant_model_path, task="detect")
    except Exception as e:
        print(f"Error during model quantization: {e}")
        return None


def extract_rois(image: np.ndarray, boxes: np.ndarray, sx: float, sy: float) -> list:
    """
    Extract ROI patches from an image using bounding boxes.
    :param image: Full-resolution BGR image.
    :param boxes: Nx4 array of [x1, y1, x2, y2] coordinates.
    :param sx: Scale factor for x-axis.
    :param sy: Scale factor for y-axis.
    :return: List of cropped ROI images.
    """
    patches = []
    boxes = sorted(boxes, key=lambda x: x[0])  # Sort by x1 coordinate
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4]*[sx, sy, sx, sy])
        # Ensure coordinates are within image bounds
        patches.append(image[y1:y2, x1:x2].copy())
    return patches


def compose_canvas(patches: list, bg_size: tuple = None) -> np.ndarray:
    """
    Compose a display canvas from ROI patches.
    If bg_size is provided, overlay patches onto a black background of given size.
    Otherwise, concatenate patches horizontally.
    :param patches: List of ROI images.
    :param bg_size: Optional (height, width) of background canvas.
    :return: Combined BGR image for display.
    """
    if not patches:
        if bg_size:
            return np.zeros((bg_size[0], bg_size[1], 3), dtype=np.uint8)
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # If single patch, return it
    if len(patches) == 1:
        return patches[0] 

    # Multiple patches: pad to same height then hstack
    heights = [p.shape[0] for p in patches]
    max_h = max(heights)
    resized = []
    for p in patches:
        h, w = p.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=p.dtype)
            p = np.vstack((p, pad))
        resized.append(p)
    return np.hstack(resized)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """
    Encode a BGR image to JPEG bytes.
    :param frame: HxWx3 BGR image.
    :param quality: JPEG quality (0-100).
    :return: Encoded JPEG byte string.
    """
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ret else b''


def decode_jpeg(data: bytes) -> np.ndarray:
    """
    Decode JPEG bytes to a BGR image.
    :param data: JPEG byte string with length prefix removed.
    :return: Decoded HxWx3 BGR image.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((1, 1, 3), dtype=np.uint8)
