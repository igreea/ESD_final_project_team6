import threading
import cv2
import torch
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
from util import capture_thread, detect_thread, display_thread

# 모델 로드 및 설정
model = YOLO("yolo11n.pt")
model.classes = [0]  # 사람 클래스만

# 해상도 설정
HIGH_RES = (1280, 720)
LOW_RES = (480, 270)

# Picamera2 초기화
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": HIGH_RES})
picam2.configure(preview_config)
picam2.start()

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
