import cv2


def find_webcam():
    # 테스트할 장치 번호의 최대값 (보통 0~10으로 테스트)
    max_tested_devices = 10

    for device_index in range(max_tested_devices):
        cap = cv2.VideoCapture(device_index)

        if cap.isOpened():
            print(f"Webcam found at index {device_index}")
            cap.release()
        else:
            print(f"No webcam found at index {device_index}")


find_webcam()
