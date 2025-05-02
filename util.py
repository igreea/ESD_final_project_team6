from picamera2 import Picamera2
import cv2
import numpy as np
import threading
from queue import Queue, Empty

# 해상도 설정
HIGH_RES = (1280, 720)
LOW_RES = (320, 180)
LOW_RES_T = (LOW_RES[1], LOW_RES[0])

# 전역 프레임 버퍼
#high_res_frame = None
#low_res_frame = None
#detections = []
hi_queue = Queue(maxsize=1)
lo_queue = Queue(maxsize=1)
det_queue = Queue(maxsize=1)

stop_event = threading.Event()

def picam2_init():
    """
    Picamera2 초기화 및 설정
    high-resolution(main)과 low-resolution(lores) 두 이미지 동시 캡쳐 
    :return: Picamera2 객체
    """
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": HIGH_RES, "format": "RGB888"},
        lores={"size": LOW_RES}
        )
    picam2.configure(preview_config)
    picam2.start()
    return picam2

def capture_thread(picam2):
    """
    카메라에서 프레임을 캡처하여 원본 프레임은 hi_queue에,
    저해상도 프레임은 lo_queue에 저장
    :param picam2: Picamera2 객체
    """
    global hi_queue, lo_queue, stop_event
    while not stop_event.is_set():
        hi = picam2.capture_array("main")
        lo = picam2.capture_array("lores")
        # 이전 프레임 버리기
        if hi_queue.full():
            hi_queue.get_nowait()
        if lo_queue.full():
            lo_queue.get_nowait()
        hi_queue.put(hi)
        lo_queue.put(lo)

def detect_thread(model):
    """
    lo_queue에서 YOLO 모델을 사용하여 사람 감지
    xyxy는 yolo에서 제공하는 attribute로, [x1, y1, x2, y2, conf, cls]를 포함
    :param model: YOLO 모델 객체
    """
    global lo_queue, det_queue, stop_event
    while not stop_event.is_set():
        try:
            frame = lo_queue.get(timeout=0.1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        except Empty:
            continue
        results = model(rgb, imgsz=LOW_RES_T, rect=True)[0]
        dets = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        if det_queue.full():
            det_queue.get_nowait()
        det_queue.put(dets)

def display_thread(picam2):
    """
    고해상도 프레임과 저해상도 프레임을 각기 다른 창에 표시
    :param picam2: Picamera2 객체
    """
    global hi_queue, lo_queue, det_queue, stop_event
    win_low = "Low-Res Stream"
    win_roi = "High-Res ROI Detection"
    #win_roi2 = "High-Res ROI Detection 2"
    cv2.namedWindow(win_low, cv2.WINDOW_AUTOSIZE)

    while not stop_event.is_set():
        try:
            hi = hi_queue.get(timeout=0.1)
            lo = lo_queue.get()
            dets = det_queue.get()
        except Empty:
            continue
            
        cv2.imshow(win_low, lo)

        if len(dets):
            dets_to_show = dets[:2] if len(dets) >= 3 else dets
            patches = []
            sx = HIGH_RES[0] / LOW_RES[0]
            sy = HIGH_RES[1] / LOW_RES[1]
            for x1, y1, x2, y2 in dets_to_show:   
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                hr1, hr2 = int(x1*sx), int(y1*sy)
                hr3, hr4 = int(x2*sx), int(y2*sy)
                roi = hi[hr2:hr4, hr1:hr3]
                if roi.size:
                    patches.append(roi)
            if len(patches) > 1:
                h1, w1 = patches[0].shape[:2]
                h2, w2 = patches[1].shape[:2]
                canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=patches[0].dtype)
                canvas[:h1, :w1] = patches[0]
                canvas[:h2, w1:w1+w2] = patches[1]
                cv2.imshow(win_roi, canvas)
            else:
                cv2.imshow(win_roi, patches[0])
        else:
            cv2.destroyWindow(win_roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    picam2.stop()
    cv2.destroyAllWindows()