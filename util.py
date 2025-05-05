from ultralytics import YOLO

def load_onnx_model(model_path):
    """
    Load the ONNX model using ONNX Runtime.
    :param model_path: Path to the ONNX model file.
    :return: ONNX Runtime session object.
    """
    try:
        model = YOLO(model_path)
        model.export(format="onnx")
        onnx_name = model_path.split(".")[0] + ".onnx"
        onnx_model = YOLO(onnx_name)
        if not onnx_model is None:
            print("ONNX model generated successfully.")
        return onnx_model
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None