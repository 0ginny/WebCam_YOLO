import cv2
import torch
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("./src/240724_yolo_crop_segmentation_pattern.pt")
model_names = model.names

# GPU 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def yolo_obj_draw(img, threshold=0.3):
    detection = model(img)[0]
    boxinfos = detection.boxes.data.tolist()

    for data in boxinfos:
        x1, y1, x2, y2 = map(int, data[:4])
        confidence_score = round(data[4], 2)
        classid = int(data[5])
        name = model_names[classid]
        if confidence_score > threshold:
            model_text = f'{name}_{confidence_score}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img,
                        text=model_text,
                        org=(x1-50, y1-50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3,
                        color=(255, 0, 0),
                        thickness=5)

# 캡쳐보드 인식
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)

# 촬영 속도 (밀리초 단위)
frame_delay = 300  # 30ms delay between frames, adjust this value to change the speed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detection = model(image_RGB)[0]
    boxinfos = detection.boxes.data.tolist()

    yolo_obj_draw(frame, threshold=0.2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
