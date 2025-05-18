High-Resolution ROI Capture by RaspberryPi4 
===========================================
> 2025-1 ESD Team 6
> 
> *아직 개발중인 프로젝트로써, 향후 변경 사항이 발생할 수 있습니다.*



## 프로젝트 개요
본 프로젝트는 Raspberry Pi 4와 YOLOv11 Nano 모델을 활용하여, PiCamera로부터 촬영된 영상을 바탕으로 저해상도 기반 ROI(Region of Interest) 크롭 스트리밍과 전체 프레임 스트리밍 방식의 비교 평가를 진행합니다. 이때 제한된 대역폭 환경을 구성하여 전체 프레임 방식 대비 ROI 크롭 방식의 전송 효율을 실험합니다.

## 주요 기능
- **실시간 객체 검출**: Pi Camera 영상에서 Fine-tuning YOLOv11 Nano 모델로 사람 객체를 검출.
- **ROI 크롭 & 전송**: 검출된 영역+저해상도 배경 JPEG 인코딩 후 TCP 소켓을 통해 전송.
- **전체 프레임 스트리밍**: JPEG 인코딩한 전체 프레임을 TCP 전송.
- **네트워크 제약 시뮬레이션**: Linux `tc` 명령어로 유선/무선 대역폭 제한 환경 구성.
- **성능 로깅**: 각 단계(Capture, Detection, Encoding, Transmission, Decoding // 변경 가능)별 FPS와 Throughput(Mbps) 기록.



## 운용 branch 현황
```
main: 메인 코드 전용, 라즈베리파이 구동 확인된 코드만 업로드

develop: 개발중인 코드, feature 추가시 해당 branch로 merge 

feature/webcam: webcam을 통한 local test용 코드

feature/onnx: ONNX 기반 모델 런타임 최적화 적용 코드

feature/LAN: 서버-클라이언트 코드

loop: legacy branch (현재 사용 안함)

```



## 설치 및 실행
1. **의존성 설치**
    ```bash
    추후 Docker 형태로 배포 예정
    ```
2. **네트워크 환경 설정**
    ```bash
    tc qdisc add dev eth0 root tbf rate [대역폭] burst [대역폭*레이턴시] latency [레이턴시]
    ```
3. **서버 및 클라이언트 동작**
    ```bash
    (라즈베리파이)python client_stream.py --[parser 입력]
    ```
    ```bash
    (노트북)python server_stream.py --[parser 입력]
    ```



## 성능 평가

* **Metrics**: 각 단계별 FPS, Throughput(Mbps), 전송량, coco 기반 mAP-객체 탐지 성능 지표
* **환경**: 유선 LAN, 제한 대역폭 1~10 Mbits
* **비교**:

  * ROI 크롭 모드 vs 전체 프레임 모드
  * 스트리밍 상황 / 내부 저장 후 전송 상황
  * Fine-tuning 전 / 후 모델 mAP

## 향후 계획

* h.264 인코딩 경로(VPU) 추가 및 비교
* 프레임 스킵 및 ROI 패킹 최적화

