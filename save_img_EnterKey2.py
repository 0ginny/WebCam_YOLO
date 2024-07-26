import cv2
import os
import tkinter as tk
from tkinter import Label, Entry, messagebox, ttk, Button
from PIL import Image, ImageTk

# 전역 변수 초기화
capture_image = None
save_dir = "./save_img"
cap = None
previous_filename = None

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def update_frame():
    global capture_image
    if cap is not None and cap.isOpened():
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

def start_webcam():
    global cap
    selected_cam = camera_combobox.get()
    if selected_cam:
        cap_index = int(selected_cam)
        cap = cv2.VideoCapture(cap_index, cv2.CAP_DSHOW)
        update_frame()

def stop_webcam():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        webcam_label.config(image='')

    control_frame.pack_forget()
    init_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

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

# 웹캠 비디오 출력 레이블
webcam_label = Label(root)
webcam_label.pack(side=tk.TOP, pady=10)

# 초기 제어 프레임
init_control_frame = tk.Frame(root)
init_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

Label(init_control_frame, text="사용할 웹캠을 선택한 후 start를 누르세요").pack(pady=5)

camera_combobox = ttk.Combobox(init_control_frame, values=find_cameras(), state="readonly")
camera_combobox.pack(pady=5)
camera_combobox.current(0)

start_button = Button(init_control_frame, text="Start", command=lambda: [init_control_frame.pack_forget(),
                                                                         control_frame.pack(side=tk.BOTTOM, fill=tk.X,
                                                                                            padx=10, pady=10),
                                                                         start_webcam()])
start_button.pack(pady=5)

# 제어 프레임
control_frame = tk.Frame(root)

instructions_label = Label(control_frame, text="저장하고 싶은 파일명을 입력한 후 엔터로 사진을 저장합니다")
instructions_label.pack(pady=5)

filename_entry = Entry(control_frame, width=50, justify='center')
filename_entry.pack(pady=5)
filename_entry.bind("<Return>", save_image)  # Enter 키를 눌렀을 때 save_image 함수 호출

stop_button = Button(control_frame, text="Stop", command=stop_webcam)
stop_button.pack(pady=5)

# 프로그램 종료 시 웹캠 자원 해제
root.protocol("WM_DELETE_WINDOW", lambda: [cap.release() if cap else None, root.destroy()])

root.mainloop()
