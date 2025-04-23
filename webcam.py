import threading
import cv2
import torch
# from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
from util import capture_thread, detect_thread, display_thread

# 모델 로드 및 설정
model = YOLO("yolo11n.pt")
model.classes = [0]  # 사람 클래스만

# 해상도 설정
HIGH_RES = (1280, 720)
LOW_RES = (480, 270)

# webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame_hd = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # ===== 실시간 출력용으로 저해상도 프레임 생성 =====
    frame_lowres = cv2.resize(frame_hd, (640, 480), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Low-Res Preview (640x480)", frame_lowres)

    # ===== YOLO에 저해상도 프레임으로 사람만 탐지 요청 =====
    results = model.predict(source=frame_lowres, classes=[0], conf=0.4, verbose=False)

    # ===== 사람 탐지된 경우 고해상도 원본 ROI 출력 =====
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 비율 변환: 저화질 → 고화질 (1:2)
        x1_hd, y1_hd, x2_hd, y2_hd = x1*2, y1*2, x2*2, y2*2

        # ROI 잘라내기
        roi_hd = frame_hd[y1_hd:y2_hd, x1_hd:x2_hd]

        # ROI가 유효할 때만 창 띄우기
        if roi_hd.size > 0:
            cv2.imshow("Detected Person ROI (HD)", roi_hd)
        break  # 첫 번째 사람만 처리 (여러 명 처리하려면 이 부분 제거)
    
    

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

'''
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
'''

