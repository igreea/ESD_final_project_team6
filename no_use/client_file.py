#!/usr/bin/env python3
import socket
import time
import os
import statistics
import argparse
import psutil
import config


def benchmark(file_path, server_ip, server_port, repeats, iface, buffer_size):
    """
    파일 기반 전송 벤치마크를 수행하고 통계 출력
    :param file_path: 전송할 파일 경로
    :param server_ip: 수신측 IP 주소
    :param server_port: 수신측 포트
    :param repeats: 반복 전송 횟수
    :param iface: 네트워크 인터페이스 이름
    :param buffer_size: 소켓 버퍼 크기 (bytes)
    :return: (file_size, avg_time, avg_throughput)
    """
    durations = []
    throughputs = []
    size = os.path.getsize(file_path)

    for i in range(repeats):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
        sock.connect((server_ip, server_port))

        # 전송 전 바이트 초기화
        tx0 = psutil.net_io_counters(pernic=True)[iface].bytes_sent

        start = time.perf_counter()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(buffer_size)
                if not chunk:
                    break
                sock.sendall(chunk)
        sock.shutdown(socket.SHUT_WR)
        end = time.perf_counter()

        # 전송 후 바이트 측정
        tx1 = psutil.net_io_counters(pernic=True)[iface].bytes_sent

        duration = end - start
        bytes_sent = tx1 - tx0
        mbps = bytes_sent * 8 / duration / 1e6

        durations.append(duration)
        throughputs.append(mbps)
        sock.close()

    print(f"=== Benchmark: {file_path} ({size} bytes) ===")
    print(f"Avg Time      : {statistics.mean(durations):.4f} s ± {statistics.stdev(durations):.4f}")
    print(f"Avg Throughput: {statistics.mean(throughputs):.2f} Mbps ± {statistics.stdev(throughputs):.2f}\n")
    return size, statistics.mean(durations), statistics.mean(throughputs)


def main():
    parser = argparse.ArgumentParser(description='File transfer benchmark')
    parser.add_argument('--server-ip',   default=config.DEFAULT_SERVER_IP,
                        help='Receiver IP address')
    parser.add_argument('--port',        type=int, default=config.SERVER_PORT,
                        help='Receiver port')
    parser.add_argument('--iface',       default=config.IFACE,
                        help='Network interface')
    parser.add_argument('--buffer-size', type=int, default=config.BUFFER_SIZE,
                        help='Socket buffer size (bytes)')
    parser.add_argument('--repeats',     type=int, default=5,
                        help='Number of repetitions')
    parser.add_argument('--file-full',   default='full_hd.bin',
                        help='Full HD file path')
    parser.add_argument('--file-roi',    default='roi.bin',
                        help='ROI file path')
    args = parser.parse_args()

    size_full, time_full, thr_full = benchmark(
        args.file_full, args.server_ip, args.port,
        args.repeats, args.iface, args.buffer_size
    )
    size_roi, time_roi, thr_roi = benchmark(
        args.file_roi, args.server_ip, args.port,
        args.repeats, args.iface, args.buffer_size
    )

    reduction = 1 - (size_roi / size_full) if size_full > 0 else 0
    speedup   = (time_full - time_roi) / time_full if time_full > 0 else 0

    print('=== Summary ===')
    print(f"Capacity Reduction : {reduction:.2%}")
    print(f"Speed Improvement  : {speedup:.2%}")
    print(f"Full Throughput    : {thr_full:.2f} Mbps")
    print(f"ROI Throughput     : {thr_roi:.2f} Mbps")


if __name__ == '__main__':
    main()
