import socket
import cv2
import numpy as np
import struct
import threading
import time
from queue import Queue, Full, Empty

from config import SERVER_HOST, SERVER_PORT, STATS_INTERVAL, BUFFER_SIZE
import util

# 수신 IP/포트 (sender 쪽 --target_ip로 지정된 값)
PORT_LO = SERVER_PORT          # 저해상도 프레임 수신 포트
PORT_HI = SERVER_PORT + 1          # 고해상도 프레임 수신 포트
PORT_LAT_SEND = SERVER_PORT + 2    # sender 쪽 --port_lat_send 로 보낸 레이턴시 패킷을 수신(=에코)할 포트
PORT_LAT_RECV = SERVER_PORT + 3    # receiver 쪽 --port_lat_recv 로 보낸 레이턴시 패킷을 수신(=에코)할 포트

# 소켓 생성 및 바인드
sock_lo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_lo.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Nagle's algorithm 비활성화
sock_lo.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 주소 재사용 허용
sock_lo.bind((SERVER_HOST, PORT_LO))
sock_lo.listen(1)
conn_lo, addr_lo = sock_lo.accept()

sock_hi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_hi.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Nagle's algorithm 비활성화
sock_hi.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 주소 재사용 허용
sock_hi.bind((SERVER_HOST, PORT_HI))
sock_hi.listen(1)
conn_hi, addr_hi = sock_hi.accept()

sock_lat = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_lat.bind((SERVER_HOST, PORT_LAT_SEND))
sock_lat.setblocking(False)  # 논블로킹 모드로 설정



# 큐 생성
send_queue_lo = Queue(maxsize=5)  # 저해상도 프레임 전송용 큐
send_queue_hi = Queue(maxsize=5)  # 고해상도 프레임 전송용 큐
save_queue_lo = Queue(maxsize=5)  # 저해상도 프레임 저장용 큐
save_queue_hi = Queue(maxsize=5)  # 고해상도 프레임 저장용 큐

stop_event = threading.Event()

def _lo_recv(conn: socket.socket):
    lo_frame_count = 0
    lo_bytes = 0
    last_stats_lo = time.perf_counter()
    while not stop_event.is_set():
        header_lo = b''
        while len(header_lo) < 4:
            chunk_header_lo = conn.recv(4 - len(header_lo))
            if not chunk_header_lo:
                # 연결 종료 신호 → 모든 쓰레드 종료
                stop_event.set()
                print("Connection closed by sender (LO) while reading header")
                return
            header_lo += chunk_header_lo
        length_lo = int.from_bytes(header_lo, 'big')

        # 페이로드 수신
        data_lo = b''
        while len(data_lo) < length_lo:
            chunk_lo = conn.recv(min(BUFFER_SIZE, length_lo - len(data_lo)))
            if not chunk_lo:
                break
            data_lo += chunk_lo
        if not data_lo:
            continue

        # JPEG 디코딩
        frame_lo = cv2.imdecode(np.frombuffer(data_lo, dtype=np.uint8), cv2.IMREAD_COLOR)
        lo_bytes += length_lo + 4
        lo_frame_count += 1

        # 통계 출력
        now_lo = time.perf_counter()
        if now_lo - last_stats_lo >= STATS_INTERVAL:
            fps = lo_frame_count / (now_lo - last_stats_lo)
            mbps = lo_bytes * 8 / (now_lo - last_stats_lo) / 1e6
            print(f"[Lo frame] {fps:.1f} FPS, {mbps:.2f} Mbps")
            lo_frame_count = 0
            lo_bytes = 0
            last_stats_lo = now_lo
        try:
            send_queue_lo.put(frame_lo, block=False)  # 논블로킹으로 큐에 프레임 추가
        except Full:
            _ = send_queue_lo.get()  # 큐가 가득 찼을 때 가장 오래된 프레임 제거
            send_queue_lo.put(frame_lo, block=False)  # 새 프레임 추가

def _hi_recv(conn: socket.socket):
    hi_frame_count = 0
    hi_bytes = 0
    last_stats_hi = time.perf_counter()
    while not stop_event.is_set():
        header_hi = b''
        while len(header_hi) < 4:
            chunk_header_hi = conn.recv(4 - len(header_hi))
            if not chunk_header_hi:
                # 연결 종료 신호 → 모든 쓰레드 종료
                stop_event.set()
                print("Connection closed by sender (HI) while reading header")
                return
            header_hi += chunk_header_hi
        length_hi = int.from_bytes(header_hi, 'big')

        # 페이로드 수신
        data_hi = b''
        while len(data_hi) < length_hi:
            chunk_hi = conn.recv(min(BUFFER_SIZE, length_hi - len(data_hi)))
            if not chunk_hi:
                break
            data_hi += chunk_hi
        if not data_hi:
            continue

        # JPEG 디코딩
        frame_hi = cv2.imdecode(np.frombuffer(data_hi, dtype=np.uint8), cv2.IMREAD_COLOR)
        hi_bytes += length_hi + 4
        hi_frame_count += 1

        # 통계 출력
        now_hi = time.perf_counter()
        if now_hi - last_stats_hi >= STATS_INTERVAL:
            fps = hi_frame_count / (now_hi - last_stats_hi)
            mbps = hi_bytes * 8 / (now_hi - last_stats_hi) / 1e6
            print(f"[Hi frame] {fps:.1f} FPS, {mbps:.2f} Mbps")
            hi_frame_count = 0
            hi_bytes = 0
            last_stats_hi = now_hi
        
        try:
            send_queue_hi.put(frame_hi, block=False)  # 논블로킹으로 큐에 프레임 추가
        except Full:
            _ = send_queue_hi.get()
            send_queue_hi.put(frame_hi, block=False)  # 새 프레임 추가

def _latency_echo(sock: socket.socket):
    while not stop_event.is_set():
        try:
            data_lat, addr_lat = sock.recvfrom(1024)
            sock.sendto(data_lat, addr_lat)  # 받은 그대로 되돌려준다
        except BlockingIOError:
            pass

def _display_frames():# OpenCV 윈도우 생성
    cv2.namedWindow("Received LO", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Received HI", cv2.WINDOW_NORMAL)
    while not stop_event.is_set():
        try:
            frame_lo = send_queue_lo.get(timeout=0.02)
            #cv2.cvtColor(frame_lo, cv2.COLOR_RGB2BGR, frame_lo)  # OpenCV는 BGR을 사용하므로 변환
            cv2.imshow("Received LO", frame_lo)
            save_queue_lo.put(frame_lo, block=True)  # 논블로킹으로 큐에 프레임 추가
        except Empty:
            pass
        
        try:
            frame_hi = send_queue_hi.get(timeout=0.02)
            #cv2.cvtColor(frame_hi, cv2.COLOR_RGB2BGR, frame_hi)  # OpenCV는 BGR을 사용하므로 변환
            cv2.imshow("Received HI", frame_hi)
            save_queue_hi.put(frame_hi, block=True)  # 논블로킹으로 큐에 프레임 추가
        except Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            cv2.destroyAllWindows()
            break

def _save_frames_to_video():
    """
    frame_lo와 frame_hi를 각각 video_lo.mp4, video_hi.mp4로 저장하는 함수.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_lo = None
    out_hi = None
    fps = 15  # 필요시 조정
    target_size = (1280, 1280)  # 필요시 조정
    hi_saving = False
    hi_blank_start = None  # blank 상태 시작 시간
    while not stop_event.is_set():
        try:
            frame_lo = save_queue_lo.get(timeout=0.02)
            if out_lo is None:
                h, w = frame_lo.shape[:2]
                out_lo = cv2.VideoWriter('video_lo.mp4', fourcc, fps, (w, h))
            out_lo.write(frame_lo)
        except Empty:
            pass

        try:
            frame_hi = save_queue_hi.get(timeout=0.02)
            frame_hi_fixed = util._resize_and_pad(frame_hi, target_size=target_size)
            is_blank = util._is_blank(frame_hi)

            now = time.time()
            if is_blank:
                if hi_saving:
                    if hi_blank_start is None:
                        hi_blank_start = now
                    elif now - hi_blank_start >= 2.0:
                        # 2초 이상 blank → 저장 중지
                        hi_saving = False
                        hi_blank_start = None
                        if out_hi is not None:
                            out_hi.release()
                            out_hi = None
                        print("[HI] 2초 이상 blank 감지, 저장 중지")
                continue  # blank면 저장하지 않음
            else:
                hi_blank_start = None
                if not hi_saving:
                    # 정상 프레임이 처음 들어오면 새 파일로 저장 시작
                    out_hi = cv2.VideoWriter(util._get_now_filename('video_hi'), fourcc, fps, target_size)
                    hi_saving = True
                    print("[HI] 정상 프레임 감지, 저장 시작")
                if out_hi is not None:
                    out_hi.write(frame_hi_fixed)
        except Empty:
            pass

        if stop_event.is_set():
            break

    if out_lo is not None:
        out_lo.release()
    if out_hi is not None:
        out_hi.release()

try:
    threads = [
        threading.Thread(target=_lo_recv, args=(conn_lo,), daemon=True),
        threading.Thread(target=_hi_recv, args=(conn_hi,), daemon=True),
        threading.Thread(target=_latency_echo, args=(sock_lat,), daemon=True),
        threading.Thread(target=_display_frames),
        threading.Thread(target=_save_frames_to_video, daemon=True)  # 비디오 저장 쓰레드 추가
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
except KeyboardInterrupt:
    stop_event.set()
    pass

cv2.destroyAllWindows()
sock_lo.close()
sock_hi.close()
sock_lat.close()