
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("./src/240724_yolo_crop_segmentation_pattern.pt")

# 글로벌 변수 초기화
cap = None
running = False

def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)  # 웹캠 시작
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam")
            return
        running = True
        thread = threading.Thread(target=update_frame)
        thread.start()

def stop_camera():
    global cap, running
    running = False
    if cap is not None:
        cap.release()  # 웹캠 해제
        cap = None
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

def update_frame():
    global cap, running
    while running:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            break

        # YOLO 모델로 객체 검출
        results = model(frame)
        # 검출된 객체의 바운딩 박스를 그리기
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{box.cls.item()} {box.conf.item():.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # OpenCV 이미지를 PIL 이미지로 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # tkinter Label 업데이트
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)

        # 10ms 대기
        if running:
            lbl_video.after(10, update_frame)

def on_closing():
    stop_camera()
    window.destroy()

# tkinter 윈도우 생성
window = tk.Tk()
window.title("YOLO 웹캠 객체 검출")
window.protocol("WM_DELETE_WINDOW", on_closing)

# tkinter 구성 요소 추가
lbl_video = tk.Label(window)
lbl_video.pack()

btn_start = tk.Button(window, text="Start", command=start_camera)
btn_start.pack(side=tk.LEFT, padx=10, pady=10)

btn_stop = tk.Button(window, text="Stop", command=stop_camera)
btn_stop.pack(side=tk.RIGHT, padx=10, pady=10)

# tkinter 메인루프 시작
window.mainloop()
