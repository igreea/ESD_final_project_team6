import os
import cv2
import numpy as np
import threading
import onnxruntime as ort
import time
import struct, socket
import argparse

from collections import deque
from queue import Queue, Empty, Full
from picamera2 import Picamera2
from ultralytics import YOLO

from config import SERVER_PORT, SERVER_IP
import util


class CameraProcessor:
    def __init__(
            self,
            model,
            mode: str = "picam",
            classes: list = [0],
            high_res: tuple = (1280, 1280),
            low_res: tuple = (192, 192),
            max_queue_size: int = 1,
            target_ip: str = "0.0.0.0",
            port_lo: int = 5000,
            port_hi: int = 5001,
            port_lat_send: int = 5002,
            port_lat_recv: int = 5003
            ):
        self.model = model
        self.mode = mode
        self.model.classes = classes

        if self.mode == "picam":
            self.picam2 = Picamera2()
            self.cfg = self.picam2.create_preview_configuration(
                main={"size": high_res, "format": "RGB888"}
            )
            self.picam2.configure(self.cfg)
        elif self.mode == "webcam":
            self.webcam = cv2.VideoCapture(0)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, high_res[0])
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, high_res[1])
        else:
            raise ValueError("Invalid mode. Choose 'picam' or 'webcam'.")

        # display 및 detect 관련 변수
        self.flg_count = 0

        self.high_res = high_res
        self.low_res = low_res
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.det_queue = Queue(maxsize=max_queue_size) 
        self.stop_event = threading.Event()

        self._inf_time = deque(maxlen=500)  # 최근 500개 프레임의 추론 시간 기록
        self.fps = 20  # 최저 FPS 설정
        self.delay = 1/self.fps # 0.05초 대기
        self.sx = high_res[0] / low_res[0] # 저해상도에서 고해상도로 변환할 때 x축 비율
        self.sy = high_res[1] / low_res[1] # 저해상도에서 고해상도로 변환할 때 y축 비율

        self.blank = np.zeros((100, 100, 3), dtype=np.uint8)  # 빈 프레임

        # 네트워크 관련 변수
        self.target_ip = target_ip
        self.port_lo = port_lo
        self.port_hi = port_hi
        self.port_lat_send = port_lat_send
        self.port_lat_recv = port_lat_recv

        # 소켓 설정
        self.sock_lo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lo.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock_lo.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_lo.connect((self.target_ip, self.port_lo))
        
        self.sock_hi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_hi.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock_hi.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_hi.connect((self.target_ip, self.port_hi))
        
        self.sock_lat = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_lat.settimeout(1.0)  # 최대 1초 대기
        self.sock_lat.bind(("0.0.0.0", self.port_lat_recv))  # 로컬 포트 바인딩

        # 통계 변수
        self.frame_count_lo = 0
        self.frame_count_hi = 0
        self.byte_count_lo = 0
        self.byte_count_hi = 0
        self.latency_list = deque(maxlen=500)  # 최근 500개 프레임의 지연 시간 기록


    def _capture_loop(self) -> None:
        """
        picamera2에서 프레임을 캡처하여 lo_queue와 hi_queue에 저장하는 스레드
        :return: None
        """
        print("start capture")
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
        print("start detect")
        while not self.stop_event.is_set():
            try:
                rgb, _ = self.frame_queue.get(timeout=0.005)
            except Empty:
                continue
            #rgb = bgr[..., ::-1]  # BGR to RGB view
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
        print("start display")
        temp_count = 0
        dets = None  # 초기값 설정
        renew_time = time.perf_counter()
        cv2.namedWindow("for quit", cv2.WINDOW_NORMAL)
        cv2.imshow("for quit", self.blank)  # 종료를 위한 빈 창 생성

        while not self.stop_event.is_set():
            try:
                lo, hi = self.frame_queue.get(timeout=self.delay)
                data_lo = util.encode_jpeg(lo, quality=ENCODE_JPEG_QUALITY)
                self.sock_lo.sendall(len(data_lo).to_bytes(4, 'big') + data_lo)
                self.frame_count_lo += 1
                self.byte_count_lo += len(data_lo) + 4  # 헤더 크기 포함
            except Empty:
                continue



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
                patches = util.extract_rois(hi, dets_to_show, self.sx, self.sy)  # util.extract_rois 사용
                canvas = util.compose_canvas(patches)  # util.compose_canvas 사용
            else:
                canvas = self.blank

            data_hi = util.encode_jpeg(canvas, quality=ENCODE_JPEG_QUALITY)
            self.sock_hi.sendall(len(data_hi).to_bytes(4, 'big') + data_hi)
            self.frame_count_hi += 1
            self.byte_count_hi += len(data_hi) + 4  # 헤더 크기 포함

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                print(f"Median Inference Time: {np.median(self._inf_time):.4f} seconds")
                data = np.array(self._inf_time)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                p95 = np.percentile(data, 95)
                iqr = q3 - q1
                print(f"IQR: {iqr:.4f}, Q1: {q1:.4f}, Q3: {q3:.4f}, P95: {p95:.4f} seconds")
                break


    def _status_loop(self):
        """
        네트워크 상태를 모니터링하는 스레드
        :return: None
        """
        print("start status")
        while not self.stop_event.is_set():
            start_time = time.perf_counter()
            time.sleep(1.0)  # 1초마다 상태 확인
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        
            # fps 계산
            fps_lo = self.frame_count_lo / elapsed_time
            fps_hi = self.frame_count_hi / elapsed_time

            # bitrate 계산 (mbps)
            bitrate_lo = (self.byte_count_lo * 8) / (elapsed_time * 1e6) 
            bitrate_hi = (self.byte_count_hi * 8) / (elapsed_time * 1e6)

            # latency 계산 (ms)
            if self.latency_list:
                latency_avg = np.mean(self.latency_list) * 1000
            else:
                latency_avg = 0.0

            print(f"Status Update: "
                  f"[Low]: {fps_lo:.2f} fps | {bitrate_lo:.2f} Mbps || "
                  f"[High]: {fps_hi:.2f} fps | {bitrate_hi:.2f} Mbps || "
                  f"Avg Latency: {latency_avg:.2f} ms")
            
            self.frame_count_lo = 0
            self.frame_count_hi = 0
            self.byte_count_lo = 0
            self.byte_count_hi = 0
            self.latency_list.clear()  # 상태 업데이트 후 지연 시간 기록 초기화

            # 소켓을 통해 지연 시간 측정
            try:
                send_ts = time.perf_counter()
                payload = struct.pack('d', send_ts)  # 타임스탬프를 바이너리로 변환
                self.sock_lat.sendto(payload, (self.target_ip, self.port_lat_send))
                data_echo, _ = self.sock_lat.recvfrom(1024)  # 최대 1024바이트 수신
                recv_ts = time.perf_counter()

                (orig_send_ts, ) = struct.unpack('d', data_echo[:8])  # 처음 8바이트를 타임스탬프로 변환
                rtt = recv_ts - orig_send_ts  
                self.latency_list.append(rtt)
            except (socket.timeout, struct.error) as e:
                pass


    def run(self):
        threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._detect_loop, daemon=True),
            threading.Thread(target=self._display_loop),
            threading.Thread(target=self._status_loop, daemon=True)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["webcam", "picam"], default="picam", help="Camera type to use")
    parser.add_argument('--high', nargs=2, type=int, default=(1280,1280), help='High-res WxH')
    parser.add_argument('--low', nargs=2, type=int, default=(192,192), help='Low-res WxH')
    parser.add_argument('--quant', action='store_true', help='Use quantized ONNX model')
    parser.add_argument('--model', type=str, default="best", help='Path to the ONNX model file')
    args = parser.parse_args()

    args.high = tuple(args.high)
    args.low = tuple(args.low)
    QUANT = args.quant

    ENCODE_JPEG_QUALITY = 90  # JPEG 인코딩 품질 설정

    print("now here")
    # 환경변수 설정__ 필요하면 잡아주기
    # os.environ["OMP_NUM_THREADS"] = "4"  # Disable OpenMP threads for ONNX Runtime
    # os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Disable OpenBLAS threads for ONNX Runtime
    # os.environ["TORCH_NUM_THREADS"] = "4"  # Disable PyTorch threads for ONNX Runtime
    model_path = os.path.join("./models", args.model)
    try:
        onnx_model = YOLO((model_path + ".onnx"), task="detect")
    except:
        onnx_model = util.load_onnx_model((model_path + ".pt"), res=args.low)

    if QUANT:
        try:
            onnx_model = YOLO("yolo11n_quant.onnx", task="detect")
        except:
            onnx_model = util.quant_onnx("yolo11n.onnx", "yolo11n_quant.onnx")

    raise ValueError("debug")
    camera_processor = CameraProcessor(
        model=onnx_model, 
        mode=args.type,
        high_res=args.high, 
        low_res=args.low, 
        classes=[0],
        target_ip=SERVER_IP,
        port_lo=SERVER_PORT,
        port_hi=SERVER_PORT + 1,
        port_lat_send=SERVER_PORT + 2,
        port_lat_recv=SERVER_PORT + 3
        )
    try:
        camera_processor.run()
    except KeyboardInterrupt:
        print(f"Interrupted by user. Stopping...")
        camera_processor.stop_event.set()
