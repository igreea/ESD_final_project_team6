High-Resolution ROI Capture by RaspberryPi4 
===========================================
> 2025-1 ESD Team 6


## 프로젝트 개요
본 프로젝트는 Raspberry Pi 4와 YOLOv11 Nano 모델을 활용하여, PiCamera로부터 촬영된 영상을 바탕으로 저해상도 기반 ROI(Region of Interest) 크롭 스트리밍과 전체 프레임 스트리밍 방식의 비교 평가를 진행합니다. 이때 제한된 대역폭 환경을 구성하여 전체 프레임 방식 대비 ROI 크롭 방식의 전송 효율을 실험합니다.


## 주요 기능
- **실시간 객체 검출**: Pi Camera 영상에서 Fine-tuning YOLOv11 Nano 모델로 사람 객체를 검출.
- **ROI 크롭 & 전송**: 검출된 영역+저해상도 배경 JPEG 인코딩 후 TCP 소켓을 통해 전송.
- **전체 프레임 스트리밍**: JPEG 인코딩한 전체 프레임을 TCP 전송.
- **네트워크 제약 시뮬레이션**: Linux `tc` 명령어로 유선/무선 대역폭 제한 환경 구성.
- **성능 로깅**: 각 단계(Capture, Detection, Encoding, Transmission, Decoding)별 FPS와 Bitrate(Mbps) 기록.
- **서버 저장**: 서버로 전송된 영상을 mp4v 인코딩 후 타임태그와 함께 자동 저장.


## 운용 branch 현황
```
main: 메인 코드 전용, 라즈베리파이 구동 확인된 코드만 업로드

develop: 개발중인 코드, feature 추가시 해당 branch로 merge 

feature/webcam: webcam을 통한 local test용 코드

feature/onnx: ONNX 기반 모델 런타임 최적화 적용 코드

feature/LAN: 서버-클라이언트 코드

loop: legacy branch (현재 사용 안함)

test/picam: 모델 성능 평가 브랜치

```


## 전체 구조

![image](https://github.com/user-attachments/assets/c57eccd4-3b68-4349-b62f-995b8a97abb2)


## 사용 방법
1.  **네트워크 설정**
    ```bash
    (라즈베리파이 및 노트북) config.py -> 사용하고자 하는 포트 설정, 클라이언트(라즈베리파이), 서버의 ip로 config 파일 변경
    ```
    
2.  **서버 및 클라이언트 동작**
    ```bash
    (라즈베리파이)python main.py --[parser 입력]
    ```
    ```bash
    (서버)python main_server.py --[parser 입력]
    ```



## 성능 평가

* **Metrics**: 기존 대비 서버 저장 용량 / 대역폭에 따른 FPS/Bitrate / 모델 별 성능 지표 / 구현 기능 별 FPS 및 CPU 사용률
* **환경**: 유선 LAN, 제한 대역폭 5~50 Mbits
* **비교**:

  * ROI 크롭 모드 vs 전체 프레임 모드
  * 사람 존재 환경 / 사람 존재하지 않는 환경
  * Fine-tuning 전 / 후 모델 mAP

## 평가 결과

1. 서버 저장량 비교

![image](https://github.com/user-attachments/assets/f521056a-ed65-43d9-b20d-3ed73e544288)
![image](https://github.com/user-attachments/assets/789970ac-f80e-460c-a5b1-7c5279d0e752)

   
<br>

2. 대역폭 별 FPS/Bitrate 비교

![image](https://github.com/user-attachments/assets/499360b0-57cd-4bd3-ba64-827aee52871a) ![image](https://github.com/user-attachments/assets/029c607c-8ae5-4d01-b4b5-6614cd60aa8e)
![image](https://github.com/user-attachments/assets/c6537225-129d-4c63-b0f8-6b588a5e9b9d)

   
<br><br>

3. 모델 별 성능 차이 비교

![image](https://github.com/user-attachments/assets/a45db6ce-2f9e-4da2-b3c1-b6ba1a2fd7b0)
![image](https://github.com/user-attachments/assets/ec487442-85f2-454c-8773-3a6214c9daeb)

   
<br><br>

4. 구현 기능 별 FPS/CPU 사용률

![image](https://github.com/user-attachments/assets/e06c59d6-06be-444a-ad4f-af37c9e53e3e)
![image](https://github.com/user-attachments/assets/e665a373-9dfa-481b-8a5c-66d944c2f362)

   

