import threading
import cv2
import torch
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
from util import capture_thread, detect_thread, display_thread, picam2_init

# 모델 로드 및 설정
model = YOLO("yolo11n.pt")
model.classes = [0]  # 사람 클래스만

# Picamera2 초기화
picam2 = picam2_init()

# 스레드 시작
threads = [
    threading.Thread(target=capture_thread(picam2= picam2), daemon=True),
    threading.Thread(target=detect_thread(model=model), daemon=True),
    threading.Thread(target=display_thread(picam2=picam2), daemon=False)
]

for t in threads:
    t.start()

for t in threads:
    t.join()
