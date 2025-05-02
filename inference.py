import os
import cv2
import numpy as np
import threading
import onnxruntime as ort
from time import time
from queue import Queue, Empty
from picamera2 import Picamera2
from ultralytics import YOLO
import util

class CameraProcessor:
    
    def __init__(self):
        pass
    def run(self):
        pass
    

if __name__ == "__main__":
    try:
        model = YOLO("yolo11n.onnx")
    except:
        model = util.load_onnx_model("yolo11n.pt")
        
    os.environ["OMP_NUM_THREADS"] = "4"  # Disable OpenMP threads for ONNX Runtime
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Disable OpenBLAS threads for ONNX Runtime
    os.environ["TORCH_NUM_THREADS"] = "4"  # Disable PyTorch threads for ONNX Runtime
    camera_processor = CameraProcessor()
    camera_processor.run()

