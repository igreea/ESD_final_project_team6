from ultralytics import YOLO
from onnxruntime.quantization import shape_inference, quantize_dynamic, QuantType

def load_onnx_model(model_path:str, res:tuple = (640, 640)) -> YOLO:
    """
    Load the ONNX model using ONNX Runtime.
    :param model_path: Path to the ONNX model file.
    :return: ONNX Runtime session object.
    """
    try:
        model = YOLO(model_path)
        model.export(format="onnx", nms=False, imgsz=res, dynamic=False, device="cpu")
        onnx_name = model_path.split(".")[0] + ".onnx"
        onnx_model = YOLO(onnx_name, task="detect")
        if not onnx_model is None:
            print("ONNX model generated successfully.")
        return onnx_model
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None
    
def quant_onnx(model_path:str, quant_model_path:str) -> YOLO:
    """
    Quantize the ONNX model using ONNX Runtime.
    :param model_path: Path to the ONNX model file.
    :param quant_model_path: Path to save the quantized model.
    return: Quantized ONNX Runtime session object.
    """
    model_path_pre = model_path.split(".")[0]+"_pre.onnx"
    try:
        shape_inference.quant_pre_process(model_path, model_path_pre)
    except Exception as e:
        print(f"Error during shape inference: {e}")
        return None
    
    try:
        quantize_dynamic(model_path_pre, quant_model_path, weight_type=QuantType.QUInt8)
        print("Model quantization successful.")
        return YOLO(quant_model_path, task="detect")
    except Exception as e:
        print(f"Error during model quantization: {e}")
        return None