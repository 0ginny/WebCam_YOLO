import cv2
import os
import tkinter as tk
from tkinter import Label, Button, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# 전역 변수 초기화
model = None
model_names = None
capture_image = None
processed_image = None
save_dir = "./save_img"
model_dir = "./src/yolo_model"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 웹캠에서 원본 이미지의 사이즈를 가져오기 위한 변수
frame_width = 640
frame_height = 480

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
            cv2.putText(img, text=model_text, org=(x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(255, 0, 0), thickness=2)

def update_frame():
    global capture_image
    ret, frame = cap.read()
    if ret:
        global frame_width, frame_height
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        webcam_label.imgtk = image_tk
        webcam_label.configure(image=image_tk)
        webcam_label.after(10, update_frame)
        capture_image = frame.copy()

def load_model():
    global model, model_names
    selected_model = model_combobox.get()
    if selected_model:
        model_path = os.path.join(model_dir, selected_model)
        model = YOLO(model_path)
        model_names = model.names
        messagebox.showinfo("Model Loaded", f"{selected_model} 모델을 불러왔습니다")
    else:
        messagebox.showwarning("No Model Selected", "모델을 선택해 주세요")

def capture():
    global capture_image, capture_label, processed_image
    if model is None:
        messagebox.showwarning("Model Not Loaded", "모델을 먼저 불러와주세요")
        return
    if capture_image is not None:
        processed_image = capture_image.copy()
        yolo_obj_draw(processed_image, threshold=0.2)
        yolo_img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        yolo_img_pil = Image.fromarray(yolo_img_rgb)
        yolo_img_tk = ImageTk.PhotoImage(yolo_img_pil)
        capture_label.imgtk = yolo_img_tk
        capture_label.configure(image=yolo_img_tk)

def save_image():
    global capture_image, processed_image
    count = len(os.listdir(save_dir)) + 1
    if processed_image is not None:
        filename = os.path.join(save_dir, f"capture_{count:03d}.jpg")
        cv2.imwrite(filename, processed_image)
        print(f"Image saved: {filename}")
    elif capture_image is not None:
        filename = os.path.join(save_dir, f"webcam_{count:03d}.jpg")
        cv2.imwrite(filename, capture_image)
        print(f"Image saved: {filename}")
    else:
        messagebox.showwarning("No Image Captured", "캡처된 이미지가 없습니다")

# 모델 파일 리스트 가져오기
model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

# GUI 생성
root = tk.Tk()
root.title("YOLO Webcam GUI")

# 전체 레이아웃을 위한 프레임 생성
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# 웹캠 및 캡처 화면 프레임 생성
webcam_frame = tk.Frame(main_frame)
webcam_frame.grid(row=0, column=0, padx=10, pady=10)

capture_frame = tk.Frame(main_frame)
capture_frame.grid(row=0, column=1, padx=10, pady=10)

# 웹캠 화면 레이블
webcam_label = Label(webcam_frame, text="Webcam", width=frame_width // 6, height=frame_height // 30, bg="gray")
webcam_label.pack()

# 웹캠 화면 아래에 모델 선택 및 로드 버튼
model_combobox = ttk.Combobox(webcam_frame, values=model_files)
model_combobox.pack(side=tk.LEFT, padx=5, pady=5)

load_button = Button(webcam_frame, text="Load Model", command=load_model)
load_button.pack(side=tk.LEFT, padx=5, pady=5)

# 캡처 화면 레이블 (초기 텍스트)
capture_label = Label(capture_frame, text="Capture", width=frame_width // 6, height=frame_height // 30, bg="gray")
capture_label.pack()

# 캡처 화면 아래에 캡처 및 저장 버튼
capture_button = Button(capture_frame, text="Capture", command=capture)
capture_button.pack(side=tk.LEFT, padx=5, pady=5)

save_button = Button(capture_frame, text="Save", command=save_image)
save_button.pack(side=tk.LEFT, padx=5, pady=5)

# 캡쳐보드 인식
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 웹캠 업데이트 시작
update_frame()

root.mainloop()

# Release the webcam and close windows when the GUI is closed
cap.release()
cv2.destroyAllWindows()
