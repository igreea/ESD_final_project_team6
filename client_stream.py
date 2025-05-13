#!/usr/bin/env python3
import socket
import threading
import argparse
import cv2
from queue import Queue, Full, Empty
from time import perf_counter, sleep
from picamera2 import Picamera2
from ultralytics import YOLO

import util
from config import DEFAULT_SERVER_IP, SERVER_PORT, JPEG_QUALITY, BUFFER_SIZE, MAX_QUEUE_SIZE, STATS_INTERVAL

class StreamClient:
    def __init__(self, server_ip, server_port, mode, quality, model_path, high_res, low_res, bg_size):
        # 네트워크 설정
        self.server_ip = server_ip
        self.server_port = server_port
        self.mode = mode
        self.quality = quality
        self.sx = high_res[0] / low_res[0] # 저해상도에서 고해상도로 변환할 때 x축 비율
        self.sy = high_res[1] / low_res[1] # 저해상도에서 고해상도로 변환할 때 y축 비율       
        self.bg_size = bg_size 
        self.send_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.stop_event = threading.Event()
        # 성능 통계
        self.stats = {
            'capture': {'count':0, 'time':0.0},
            'detect': {'count':0, 'time':0.0},
            'send':    {'count':0, 'time':0.0},
        }
        # 모델 로드
        try:
            self.model = YOLO(model_path, task='detect')
        except Exception:
            model_path = model_path.replace('.onnx', '.pt')
            self.model = util.load_onnx_model(model_path, res=tuple(low_res))
        # 출력 창 설정
        if self.mode == 'full':
            cv2.namedWindow('Local', cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow('Local', cv2.WINDOW_NORMAL)
            cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        # 카메라 설정
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(
            main={'size': tuple(high_res), 'format':'BGR888'},
            lores={'size': tuple(low_res), 'format':'YUV420'}
        )
        self.picam2.configure(cfg)
        self.picam2.start()

    def _capture_detect_loop(self):
        """캡처→검출→로컬 표시 및 전송 큐 enqueue"""
        while not self.stop_event.is_set():
            t0 = perf_counter()
            lores = self.picam2.capture_array('lores')
            hi_bgr = self.picam2.capture_array('main')
            # lores를 RGB로 변환
            lores = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
            t1 = perf_counter()
            # 검출
            if self.mode == 'roi':
                results = self.model(lores[..., ::-1], imgsz=lores.shape[:2])
                dets = results[0].boxes.xyxy.cpu().numpy()
            t2 = perf_counter()
            # ROI 처리
            frame = hi_bgr if self.mode=='full' else util.compose_canvas(
                util.extract_rois(hi_bgr, dets, self.sx, self.sy), bg_size=self.bg_size)
            
            # 로컬 표시
            if self.mode == 'full':
                cv2.imshow('Local', frame)
            else:
                cv2.imshow('Local', lores)
                cv2.imshow('ROI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break
            # 큐에 전송될 프레임
            try:
                self.send_queue.put(frame, block=False)
            except Full:
                pass
            # 통계 업데이트
            self.stats['capture']['count'] += 1
            self.stats['capture']['time']  += (t1 - t0)
            if self.mode == 'roi':
                self.stats['detect']['count']  += 1
                self.stats['detect']['time']   += (t2 - t1)

    def _sender_loop(self):
        """큐에서 프레임을 꺼내 JPEG 인코딩 후 서버로 전송"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((self.server_ip, self.server_port))
        while not self.stop_event.is_set():
            try:
                frame = self.send_queue.get(timeout=0.05)
            except Empty:
                continue
            t0 = perf_counter()
            data = util.encode_jpeg(frame, quality=self.quality)
            sock.sendall(len(data).to_bytes(4, 'big') + data)
            t1 = perf_counter()
            # 통계 업데이트
            self.stats['send']['count'] += 1
            self.stats['send']['time']  += (t1 - t0)
        sock.close()

    def _stats_printer(self):
        """주기적으로 단계별 FPS 출력"""
        while not self.stop_event.is_set():
            sleep(STATS_INTERVAL)
            print(f"\n--- Stats ({STATS_INTERVAL}s) ---")
            for k,v in self.stats.items():
                fps = v['count']/v['time'] if v['time']>0 else 0
                print(f"{k:7s}: {fps:.1f} FPS ({v['count']} frames)")
                v['count'] = 0
                v['time']  = 0.0 # 통계 초기화
            print('-'*30)

    def run(self):
        threads = []
        # 캡처+검출 스레드
        t_cd = threading.Thread(target=self._capture_detect_loop)
        threads.append(t_cd)
        # 전송 스레드 (daemon)
        t_snd = threading.Thread(target=self._sender_loop, daemon=True)
        threads.append(t_snd)
        # 통계 스레드 (daemon)
        t_stat = threading.Thread(target=self._stats_printer, daemon=True)
        threads.append(t_stat)
        # 스레드 시작
        for t in threads:
            t.start()
        # 캡처+검출 스레드 종료 대기
        t_cd.join()
        # 정리
        self.stop_event.set()
        cv2.destroyAllWindows()
        self.picam2.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-ip', default=DEFAULT_SERVER_IP, help='Server IP address')
    parser.add_argument('--server-port', type=int, default=SERVER_PORT, help='Server port')
    parser.add_argument('--mode', choices=['full','roi'], required=True, help='Streaming mode')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY, help='JPEG quality')
    parser.add_argument('--model', default='yolo11n.onnx', help='Model path')
    parser.add_argument('--high', nargs=2, type=int, default=[1920,1920], help='High-res WxH')
    parser.add_argument('--low', nargs=2, type=int, default=[192,192], help='Low-res WxH')
    parser.add_argument('--bg', default=(100,100), type=int, nargs=2, help='Background WxH for ROI mode')
    args = parser.parse_args()

    client = StreamClient(
        server_ip  = args.server_ip,
        server_port= args.server_port,
        mode       = args.mode,
        quality    = args.quality,
        model_path = args.model,
        high_res   = args.high,
        low_res    = args.low,
        bg_size    = args.bg
    )
    client.run()
