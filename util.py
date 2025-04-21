from picamera2 import Picamera2
import cv2
import numpy as np

# 해상도 설정
HIGH_RES = (1280, 720)
LOW_RES = (480, 270)

# 전역 프레임 버퍼
high_res_frame = None
low_res_frame = None
detections = []

def capture_thread(picam2):
    """
    카메라에서 프레임을 캡처하여 원본 프레임은 high_res_frame에,
    저해상도 프레임은 low_res_frame에 저장
    :param picam2: Picamera2 객체
    """
    global high_res_frame, low_res_frame
    while True:
        frame = picam2.capture_array()
        high_res_frame = frame
        low_res_frame = cv2.resize(frame, LOW_RES, interpolation=cv2.INTER_AREA)

def detect_thread(model):
    """
    low_res_frame에서 YOLO 모델을 사용하여 사람 감지
    xyxy는 yolo에서 제공하는 attribute로, [x1, y1, x2, y2, conf, cls]를 포함
    :param model: YOLO 모델 객체
    """
    global low_res_frame, detections
    while True:
        if low_res_frame is None:
            continue
        results = model(low_res_frame)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

def display_thread(picam2):
    """
    고해상도 프레임과 저해상도 프레임을 각기 다른 창에 표시
    :param picam2: Picamera2 객체
    """
    global high_res_frame, low_res_frame, detections
    win_low = "Low-Res Stream"
    win_roi = "High-Res ROI Detection"
    cv2.namedWindow(win_low, cv2.WINDOW_AUTOSIZE)

    while True:
        if low_res_frame is not None:
            cv2.imshow(win_low, low_res_frame)

        # 사람 감지 시에만 고해상도 창 표시
        if high_res_frame is not None and len(detections) > 0:
            # Limit ROI count
            dets_to_show = detections[:2] if len(detections) >= 3 else detections

            roi_patches = []
            scale_x = HIGH_RES[0] / LOW_RES[0]
            scale_y = HIGH_RES[1] / LOW_RES[1]
            for *box, conf, cls in dets_to_show:
                x1, y1, x2, y2 = map(int, box)
                hr_x1, hr_y1 = int(x1 * scale_x), int(y1 * scale_y)
                hr_x2, hr_y2 = int(x2 * scale_x), int(y2 * scale_y)
                roi = high_res_frame[hr_y1:hr_y2, hr_x1:hr_x2]
                if roi.size:
                    roi_patches.append(roi)

            # Show only if any ROI patches exist
            if roi_patches:
                display_roi = np.vstack(roi_patches)
                cv2.imshow(win_roi, display_roi)
        else:
            # Close the ROI window if no detection
            cv2.destroyWindow(win_roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()