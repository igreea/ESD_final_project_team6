from ultralytics import YOLO
model_path = "best.pt"
res = (192,192)
try:
	model = YOLO(model_path)
	model.export(format="onnx", nms=False, imgsz=res, dynamic=False, device="cpu")
	onnx_name = model_path.split(".")[0] + ".onnx"
	onnx_model = YOLO(onnx_name, task="detect")
	if not onnx_model is None:
		print("ONNX model generated successfully.")
	
except Exception as e:
	print(f"Error loading ONNX model: {e}")
	
