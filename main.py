import os
import cv2
import numpy as np
import threading
import onnxruntime as ort
import time
from queue import Queue, Empty, Full
from picamera2 import Picamera2
from ultralytics import YOLO
import util
from collections import deque
import argparse

class CameraProcessor:
    def __init__(
            self,
            model,
            mode: str = "picam",
            classes: list = [0],
            high_res: tuple = (1280, 1280),
            low_res: tuple = (192, 192),
            max_queue_size: int = 1,
            ):
        self.model = model
        self.mode = mode
        self.model.classes = classes

        if self.mode == "picam":
            self.picam2 = Picamera2()
            self.cfg = self.picam2.create_preview_configuration(
                main={"size": high_res, "format": "BGR888"}
            )
            self.picam2.configure(self.cfg)
        elif self.mode == "webcam":
            self.webcam = cv2.VideoCapture(0)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, high_res[0])
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, high_res[1])
        else:
            raise ValueError("Invalid mode. Choose 'picam' or 'webcam'.")

        self.flg_count = 0

        self.high_res = high_res
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
        if self.mode == "picam":
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
        elif self.mode == "webcam":
            while not self.stop_event.is_set():
                ret, frame = self.webcam.read()
                if not ret:
                    continue
                high = frame
                low = cv2.resize(
                    src=high, 
                    dsize=self.low_res, 
                    interpolation=cv2.INTER_AREA
                )
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put((low, high), block=False)
            self.webcam.release()
        else: # 잘못된 모드 처리
            raise ValueError("Invalid mode. Choose 'picam' or 'webcam'.")
    
        if self.flg_count > 100:
            print("flag count exceeded 100, cut the capture loop.")
            self.stop_event.set()
    

    def _detect_loop(self) -> None:
        """
        lo_queue에서 프레임을 가져와 YOLO 모델로 감지하고,
        감지된 결과를 det_queue에 저장하는 스레드
        최악의 경우 timeout 지연 5ms 발생 가능
        :return: None
        """
        while not self.stop_event.is_set():
            try:
                bgr, _ = self.frame_queue.get(timeout=0.005)
            except Empty:
                continue
            rgb = bgr[..., ::-1]  # BGR to RGB view
            results = self.model(rgb, imgsz=self.low_res, verbose=False)

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
            if self.flg_count > 1000:
                print("flag count exceeded 100, cut the detect loop.")
                self.stop_event.set()
                raise RuntimeError("Flag count exceeded 1000, possible infinite loop detected.")

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
        dets = None  # 초기값 설정
        disp_time = time.perf_counter()
        renew_time = time.perf_counter()
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
                renew_end_time = time.perf_counter()
                self._inf_time.append(renew_end_time - renew_time)  # 추론 시간 기록
                renew_time = renew_end_time


            # Draw bounding boxes
            if dets is not None and len(dets) > 0:
                dets_to_show = dets[:2] if len(dets) > 2 else dets
                dets_to_show = sorted(dets_to_show, key=lambda x: x[0]) # detection 값이 sort되어야 안정적으로 ROI 출력 가능
                patches = []
                for box in dets_to_show:
                    x1, y1, x2, y2 = map(int, box[:4])
                    hr1, hr2 = int(x1*self.sx*0.95), int(y1*self.sy*0.90)
                    hr3, hr4 = int(x2*self.sx*1.05), int(y2*self.sy)
                    hr1, hr2 = max(0, hr1), max(0, hr2)
                    hr3, hr4 = min(self.high_res[0], hr3), min(self.high_res[1], hr4)
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
                elif len(patches) == 1:
                    cv2.imshow(win_hi, patches[0])
                else:
                    cv2.imshow(win_hi, self.blank)
            else:
                cv2.imshow(win_hi, self.blank)  # 빈 프레임 표시

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                print("Stopping camera processor...")
                print(f"Average Inference Time: {np.median(self._inf_time):.4f} seconds")
                data = np.array(self._inf_time)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                p95 = np.percentile(data, 95)
                iqr = q3 - q1
                print(f"IQR: {iqr:.4f}, Q1: {q1:.4f}, Q3: {q3:.4f}, P95: {p95:.4f} seconds")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["webcam", "picam"], default="picam", help="Camera type to use")
    parser.add_argument('--high', nargs=2, type=int, default=(1920,1920), help='High-res WxH')
    parser.add_argument('--low', nargs=2, type=int, default=(192,192), help='Low-res WxH')
    parser.add_argument('--quant', action='store_true', help='Use quantized ONNX model')
    args = parser.parse_args()

    args.high = tuple(args.high)
    args.low = tuple(args.low)
    QUANT = args.quant

    print("now here")
    # 환경변수 설정__ 필요하면 잡아주기
    # os.environ["OMP_NUM_THREADS"] = "4"  # Disable OpenMP threads for ONNX Runtime
    # os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Disable OpenBLAS threads for ONNX Runtime
    # os.environ["TORCH_NUM_THREADS"] = "4"  # Disable PyTorch threads for ONNX Runtime

    try:
        onnx_model = YOLO("best.onnx", task="detect")
    except:
        onnx_model = util.load_onnx_model("best.pt", res=args.low)

    if QUANT:
        try:
            onnx_model = YOLO("yolo11n_quant.onnx", task="detect")
        except:
            onnx_model = util.quant_onnx("yolo11n.onnx", "yolo11n_quant.onnx")
    
    camera_processor = CameraProcessor(
        model=onnx_model, 
        mode=args.type,
        high_res=args.high, 
        low_res=args.low, 
        classes=[0])
    try:
        camera_processor.run()
    except KeyboardInterrupt:
        print(f"Interrupted by user. Stopping...")
        camera_processor.stop_event.set()
