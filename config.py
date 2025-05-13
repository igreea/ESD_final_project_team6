# 설정 파일

# 네트워크
SERVER_HOST      = '0.0.0.0'         # 수신측 바인딩 호스트
SERVER_PORT      = 9000              # 수신측 포트
DEFAULT_SERVER_IP= '192.168.0.50'    # 송신측 기본 PC IP
IFACE            = 'eth0'            # 측정 인터페이스 이름

# JPEG 인코딩
JPEG_QUALITY     = 85                # imencode 품질 (0-100)
BUFFER_SIZE      = 64 * 1024         # 소켓 읽기/쓰기 버퍼 크기

# 큐, 반복
MAX_QUEUE_SIZE   = 2                 # 전송용 큐 최대 크기
STATS_INTERVAL   = 5.0               # 통계 출력 간격 (초)
REPEATS          = 5                 # 파일 전송 벤치마크 반복 횟수
