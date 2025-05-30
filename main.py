import os
import cv2
import numpy as np
import threading
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import time
from queue import Queue, Empty, Full
from picamera2 import Picamera2
from ultralytics import YOLO
import util
from collections import deque

class CameraProcessor:
    def __init__(
            self,
            model,
            classes: list = [0],
            high_res: tuple = (1280, 1280),
            low_res: tuple = (192, 192),
            max_queue_size: int = 1,
            ):
        self.model = model
        self.model.classes = classes
        self.picam2 = Picamera2()
        self.cfg = self.picam2.create_preview_configuration(
            main={"size": high_res, "format": "BGR888"}
        )
        self.picam2.configure(self.cfg)
        
        self.flg_count = 0

        self.low_res = low_res
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.det_queue = Queue(maxsize=max_queue_size) 
        self.stop_event = threading.Event()

        self._inf_time = deque(maxlen=500)  # 최근 500개 프레임의 추론 시간 기록
        self._disp_time = deque(maxlen=500)  # 최근 500개 프레임의 디스플레이 시간 기록
        self.fps = 20  # 최저 FPS 설정
        self.delay = 1/self.fps # 0.05초 대기
        self.sx = high_res[0] / low_res[0] # 저해상도에서 고해상도로 변환할 때 x축 비율
        self.sy = high_res[1] / low_res[1] # 저해상도에서 고해상도로 변환할 때 y축 비율

        self.blank = np.zeros((100, 100, 3), dtype=np.uint8)  # 빈 프레임


    def _capture_loop(self) -> None:
        """
        picamera2에서 프레임을 캡처하여 lo_queue와 hi_queue에 저장하는 스레드
        :return: None
        """
        self.picam2.start()
        while not self.stop_event.is_set():
            high = self.picam2.capture_array("main")
            low = cv2.resize(
                src=high, 
                dsize=self.low_res, 
                interpolation=cv2.INTER_AREA
            )
            if self.frame_queue.full():
                self.frame_queue.get_nowait()  # 큐가 가득 찼을 때 가장 오래된 프레임 제거
            self.frame_queue.put((low, high), block=False)
        self.picam2.stop()
    

    def _detect_loop(self) -> None:
        """
        lo_queue에서 프레임을 가져와 YOLO 모델로 감지하고,
        감지된 결과를 det_queue에 저장하는 스레드
        최악의 경우 timeout 지연 5ms 발생 가능
        :return: None
        """
        inf_time = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                bgr, _ = self.frame_queue.get(timeout=0.005)
            except Empty:
                continue
            rgb = bgr[..., ::-1]  # BGR to RGB view
            results = self.model(rgb, imgsz=self.low_res)
            inf_end_time = time.perf_counter()
            self._inf_time.append(inf_end_time - inf_time) # inference time record
            inf_time = inf_end_time

            dets = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            try:
                self.det_queue.get_nowait()
            except Empty:
                pass
            try:
                self.det_queue.put(dets, block=False)  # 외부 호출 없으면 FULL 발생 여지 없음
            except Full:
                pass # 사실 없어도 되는데 안전장치
            self.flg_count += 1 # 바운딩 박스 갱신 플래그

    def _display_loop(self) -> None:
        """
        고해상도 프레임과 저해상도 프레임을 각기 다른 창에 표시하는 스레드
        모델 추론이 완료될 때만 바운딩 박스 갱신, 추론 중일때는 이전 바운딩 박스 유지
        :return: None
        """
        win_hi = "High-Res"
        win_lo = "Low-Res"
        cv2.namedWindow(win_hi, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_lo, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_lo, 640, 640) #640x640으로 크기 조정
        
        temp_count = 0
        disp_time = time.perf_counter()
        while not self.stop_event.is_set():
            # Show low-res preview for debugging
            try:
                lo, hi = self.frame_queue.get(timeout=self.delay)
                # Empty 발생 시 아래는 실행되지 않음
                cv2.imshow(win_lo, lo)
                disp_end_time = time.perf_counter()
                self._disp_time.append(disp_end_time - disp_time)
                disp_time = disp_end_time
            except Empty:
                pass

            # flag가 변경되었을 때만 바운딩 박스 갱신
            if self.flg_count != temp_count:
                temp_count = self.flg_count
                try:
                    dets = self.det_queue.get_nowait()
                except Empty:
                    dets = None # 감지된 객체가 없을 경우 None으로 설정

            # Draw bounding boxes
            if dets is not None and len(dets) > 0:
                dets_to_show = dets[:2] if len(dets) > 2 else dets
                dets_to_show = sorted(dets_to_show, key=lambda x: x[0]) # detection 값이 sort되어야 안정적으로 ROI 출력 가능
                patches = []
                for box in dets_to_show:
                    x1, y1, x2, y2 = map(int, box[:4])
                    hr1, hr2 = int(x1*self.sx), int(y1*self.sy)
                    hr3, hr4 = int(x2*self.sx), int(y2*self.sy)
                    roi = hi[hr2:hr4, hr1:hr3]
                    if roi.size:
                        patches.append(roi)
                if len(patches) > 1:
                    h1, w1 = patches[0].shape[:2]
                    h2, w2 = patches[1].shape[:2]
                    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=patches[0].dtype)
                    canvas[:h1, :w1] = patches[0]
                    canvas[:h2, w1:w1+w2] = patches[1]
                    cv2.imshow(win_hi, canvas)
                else:
                    cv2.imshow(win_hi, patches[0])
            else:
                cv2.imshow(win_hi, self.blank)  # 빈 프레임 표시

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                print("Stopping camera processor...")
                print(f"Average Inference Time: {np.mean(self._inf_time):.4f} seconds")
                print(f"Average FPS: {1/(np.mean(self._disp_time)):.4f} fps")
                cv2.destroyAllWindows()
                break


    def run(self):
        threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._detect_loop, daemon=True),
            threading.Thread(target=self._display_loop)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    

if __name__ == "__main__":
    HIGH_RES = (1280, 1280)  # 고해상도 해상도
    LOW_RES = (192, 192)  # 저해상도 해상도
    QUANT = False  # 양자화 여부

    # 환경변수 설정__ 필요하면 잡아주기
    # os.environ["OMP_NUM_THREADS"] = "4"  # Disable OpenMP threads for ONNX Runtime
    # os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Disable OpenBLAS threads for ONNX Runtime
    # os.environ["TORCH_NUM_THREADS"] = "4"  # Disable PyTorch threads for ONNX Runtime

    try:
        onnx_model = YOLO("yolo11n.onnx", task="detect")
    except:
        onnx_model = util.load_onnx_model("yolo11n.pt", res=LOW_RES)

    if QUANT:
        try:
            onnx_model = YOLO("yolo11n_quant.onnx", task="detect")
        except:
            onnx_model = util.quant_onnx("yolo11n.onnx", "yolo11n_quant.onnx")
    
    camera_processor = CameraProcessor(
        model=onnx_model, 
        high_res=HIGH_RES, 
        low_res=LOW_RES, 
        classes=[0])
    camera_processor.run()

