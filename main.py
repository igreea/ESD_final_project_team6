import threading
import cv2
import torch
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
from util import *

# 모델 로드 및 설정
model = YOLO("yolo11n.pt")
model.classes = [0]  # 사람 클래스만

# 해상도 설정
HIGH_RES = (1280, 720)
LOW_RES = (480, 270)

'''
# Picamera2 초기화
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": HIGH_RES})
picam2.configure(preview_config)
picam2.start()
'''

# webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_RES[0])
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_RES[1])

# 스레드 시작
threads = [
    threading.Thread(target=capture_thread_webcam, args=(webcam,), daemon=True),
    threading.Thread(target=detect_thread, args=(model,), daemon=True),
    threading.Thread(target=display_thread, args=(webcam,), daemon=False)
]

for t in threads:
    t.start()

for t in threads:
    t.join()
