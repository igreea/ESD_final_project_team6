#!/usr/bin/env python3
import socket
import threading
import time
import psutil
import cv2
import argparse

import util
from config import SERVER_HOST, SERVER_PORT, IFACE, STATS_INTERVAL, BUFFER_SIZE


def handle_client(conn, display):
    total_bytes = 0
    frame_count = 0
    last_stats = time.time()

    if display:
        cv2.namedWindow('Remote', cv2.WINDOW_NORMAL)

    while True:
        # 4바이트 길이 헤더 읽기
        header = conn.recv(4)
        if not header:
            break
        length = int.from_bytes(header, 'big')

        # 페이로드 수신
        data = b''
        while len(data) < length:
            chunk = conn.recv(min(BUFFER_SIZE, length - len(data)))
            if not chunk:
                break
            data += chunk
        if not data:
            break

        # JPEG 디코딩
        frame = util.decode_jpeg(data)
        total_bytes += length + 4
        frame_count += 1

        # 디스플레이
        if display:
            cv2.imshow('Remote', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 통계 출력
        now = time.time()
        if now - last_stats >= STATS_INTERVAL:
            fps = frame_count / (now - last_stats)
            mbps = total_bytes * 8 / (now - last_stats) / 1e6
            print(f"[Server {IFACE}] {fps:.1f} FPS, {mbps:.2f} Mbps")
            frame_count = 0
            total_bytes = 0
            last_stats = now

    conn.close()
    if display:
        cv2.destroyAllWindows()


def run_server(host, port, display):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    print(f"Listening on {host}:{port}...")

    conn, addr = srv.accept()
    print("Client connected:", addr)
    handle_client(conn, display)
    srv.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for real-time streaming over LAN')
    parser.add_argument('--host', default=SERVER_HOST, help='Binding host')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='Port to listen on')
    parser.add_argument('--no-display', action='store_true', help='Disable frame display')
    args = parser.parse_args()

    run_server(args.host, args.port, not args.no_display)
