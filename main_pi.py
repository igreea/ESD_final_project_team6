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

'''

'''

class CameraProcessor:
    def __init__(
            self,
            model,
            classes: list = [0],
            high_res: tuple = (1280, 1280),
            low_res: tuple = (192, 192),
            max_queue_size: int = 10,
            skip_rate: int = 10     # n번마다 추론
            ):
        self.model = model
        self.model.classes = classes

        self.high_res = high_res
        self.low_res = low_res
        
        self.skip_rate = skip_rate
        self.skipping = 0
        '''
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, high_res[0])
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, high_res[1])
        '''
        self.picam2 = Picamera2()
        self.cfg = self.picam2.create_preview_configuration(
            main={"size": high_res, "format": "BGR888"},
            lores={"size": low_res, "format": "YUV420"}
        )
        self.picam2.configure(self.cfg)
        

        self.max_queue_size = max_queue_size
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.det_queue = Queue(maxsize=max_queue_size) 
        self.stop_event = threading.Event()

        self.sx = high_res[0] / low_res[0] # 저해상도에서 고해상도로 변환할 때 x축 비율
        self.sy = high_res[1] / low_res[1] # 저해상도에서 고해상도로 변환할 때 y축 비율
        self.blank = np.zeros((640, 640, 3), dtype=np.uint8)  # 빈 프레임

        # 추론시간, fps측정
        self.fps_list = []
        self.inference_times = []

    def _capture_loop(self) -> None:
        """
        picamera2에서 프레임을 캡처하여 lo_queue와 hi_queue에 저장하는 스레드
        :return: None
        """

        self.picam2.start()
        while not self.stop_event.is_set():
            low = self.picam2.capture_array("lores")
            high = self.picam2.capture_array("main")
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put((low, high), block=False)
        self.picam2.stop()
    
    def _detect_loop(self) -> None:
        #frame_count = 0
        #raise ValueError
        while not self.stop_event.is_set():
            #frame_count += 1
            #if frame_count % self.skip_rate != 0:
                
            #self.skipping = True
                #continue
            #print("$", frame_count)
            try:
                yuv, _ = self.frame_queue.get(timeout=0.05)
            except Empty:
                continue
            
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            rgb = bgr[..., ::-1]  # BGR to RGB view
            results = self.model(rgb, imgsz=self.low_res, verbose=False)
            self.skipping = self.skipping + 1
            #print(self.skipping)

            inference_time = results[0].speed["inference"]  # YOLOv8/v11은 리스트 반환
            self.inference_times.append(inference_time)
            dets = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            
            try:
                self.det_queue.get_nowait()
            except Empty:
                pass
            try:
                self.det_queue.put(dets, block=False)  # 외부 호출 없으면 FULL 발생 여지 없음
            except Full:
                pass # 사실 없어도 되는데 안전장치

    def _display_loop(self) -> None:
        #raise ValueError
        win_hi = "High-Res"
        win_lo = "Low-Res"
        cv2.namedWindow(win_hi, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_lo, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_lo, 640, 640) #640x640으로 크기 조정    
        temp = 0
        dets = None
        st_time = time.perf_counter()
        time_empty = time.perf_counter()
        while not self.stop_event.is_set():
            time_00 = time.perf_counter()
            #print(time_00-time_empty)
            # 프레임 먼저   -> blocking 제거
            try:
                lo, hi = self.frame_queue.get(timeout=0.05)
                lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)
                cv2.imshow(win_lo, lo_bgr)
                
                end_time = time.perf_counter()
                fps = 1.0/(end_time - st_time)
                print(fps)
                self.fps_list.append(fps)
                st_time = end_time
        
            except Empty:   # 비어있으면 det 볼 필요도 없음
                print("Empty!!")
                time_empty = time.perf_counter()
                continue

            # skip 중일 때만 이전 박스 유지
            if self.skipping != temp:
                temp = self.skipping
                try:
                    dets= self.det_queue.get_nowait()     # 추론 기다리기
                except Empty:       # det_queue가 비어있으면 추론된게 없는거
                    dets = None
                    pass  

            time_11 = time.perf_counter()
        
            #print(f"0011 = {time_11-time_00}")
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
                print(f"평균 fps: {sum(self.fps_list)/len(self.fps_list):.2f} ")
                print(f"평균 Inference Time: {sum(self.inference_times) / len(self.inference_times):.2f} ms")
                print("!", self.fps_list )
                print("!!", self.inference_times)
                break
            time_22 = time.perf_counter()
        
            #print(f"1122 = {time_22-time_11}")        
            #print(f"0022 = {time_22-time_00}")
            print(f"0022 fps = {1/(time_22-time_00)}")
        cv2.destroyAllWindows()

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
    ONNX = False
    
    if not ONNX:
        model = YOLO("yolo11n.pt")
    else:
        model = YOLO("yolo11n.onnx", task="detect")
    
    if QUANT:
        model = util.quant_onnx("yolo11n.onnx", "yolo11n_quant.onnx")
    
    camera_processor = CameraProcessor(model, high_res=HIGH_RES, low_res=LOW_RES, classes=[0], skip_rate=100000)
    camera_processor.run()

