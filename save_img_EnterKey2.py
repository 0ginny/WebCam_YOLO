import cv2
import os
import threading
import tkinter as tk
from tkinter import Label, Entry, messagebox, ttk, Button
from PIL import Image, ImageTk

# 전역 변수 초기화
capture_image = None
save_dir = "./save_img"
cap = None
previous_filename = None
current_camera_index = 0
lock = threading.Lock()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def update_frame():
    global capture_image, cap
    with lock:
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image_tk = ImageTk.PhotoImage(image)
                webcam_label.imgtk = image_tk
                webcam_label.configure(image=image_tk)
                capture_image = frame.copy()
    webcam_label.after(10, update_frame)


def save_image(event=None):
    global capture_image, previous_filename
    filename = filename_entry.get().strip()
    if not filename:
        filename = "img"

    existing_files = [f for f in os.listdir(save_dir) if f.startswith(filename) and f.endswith('.jpg')]
    if filename != previous_filename:
        previous_filename = filename
        if existing_files:
            existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            count = max(existing_indices) + 1
        else:
            count = 1
    else:
        if existing_files:
            existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            count = max(existing_indices) + 1
        else:
            count = 1

    file_path = os.path.join(save_dir, f"{filename}_{count:03d}.jpg")
    if capture_image is not None:
        cv2.imwrite(file_path, capture_image)
        print(f"Image saved: {file_path}")
    else:
        messagebox.showwarning("No Image Captured", "캡처된 이미지가 없습니다")


def change_camera():
    global current_camera_index
    selected_camera = int(camera_combobox.get())
    if selected_camera != current_camera_index:
        current_camera_index = selected_camera
        root.after(0, init_camera, selected_camera)


def init_camera(camera_index):
    global cap
    with lock:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    update_frame()


def find_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


# GUI 생성
root = tk.Tk()
root.title("Webcam Capture")

webcam_label = Label(root)
webcam_label.pack(side=tk.TOP)

control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

instructions_label = Label(control_frame, text="저장하고 싶은 파일명을 입력한 후 엔터로 사진을 저장합니다")
instructions_label.grid(row=0, column=0, padx=5, pady=5)

filename_entry = Entry(control_frame, width=20)
filename_entry.grid(row=1, column=0, padx=5, pady=5)
filename_entry.bind("<Return>", save_image)  # Enter 키를 눌렀을 때 save_image 함수 호출

camera_combobox = ttk.Combobox(control_frame, values=find_cameras(), state="readonly")
camera_combobox.grid(row=2, column=0, padx=5, pady=5)
camera_combobox.current(0)

change_camera_button = Button(control_frame, text="Change Camera", command=change_camera)
change_camera_button.grid(row=3, column=0, padx=5, pady=5)

# 기본 웹캠 초기화
root.after(0, init_camera, int(camera_combobox.get()))

root.mainloop()

# Release the webcam and close windows when the GUI is closed
if cap is not None:
    cap.release()
cv2.destroyAllWindows()

