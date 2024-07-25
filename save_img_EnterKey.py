import cv2
import os
import tkinter as tk
from tkinter import Label, Entry, messagebox
from PIL import Image, ImageTk

# 전역 변수 초기화
capture_image = None
save_dir = "./save_img"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def update_frame():
    global capture_image
    ret, frame = cap.read()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        webcam_label.imgtk = image_tk
        webcam_label.configure(image=image_tk)
        webcam_label.after(10, update_frame)
        capture_image = frame.copy()

def save_image(event=None):
    global capture_image
    filename = filename_entry.get().strip()
    if not filename:
        filename = "img"
    count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')]) + 1
    file_path = os.path.join(save_dir, f"{filename}_{count:03d}.jpg")
    if capture_image is not None:
        cv2.imwrite(file_path, capture_image)
        print(f"Image saved: {file_path}")
    else:
        messagebox.showwarning("No Image Captured", "캡처된 이미지가 없습니다")

# GUI 생성
root = tk.Tk()
root.title("Webcam Capture")

webcam_label = Label(root)
webcam_label.pack(side=tk.TOP)

control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

instructions_label = Label(control_frame, text="저장하고 싶은 파일명을 입력한 후 엔터로 사진을 저장합니다")
instructions_label.grid(row=0, column=0, padx=5, pady=5)

filename_entry = Entry(control_frame, width=50)
filename_entry.grid(row=1, column=0, padx=5, pady=5)
filename_entry.bind("<Return>", save_image)  # Enter 키를 눌렀을 때 save_image 함수 호출

# 웹캠 초기화
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 웹캠 인덱스가 0인지 확인

update_frame()
root.mainloop()

# Release the webcam and close windows when the GUI is closed
cap.release()
cv2.destroyAllWindows()
