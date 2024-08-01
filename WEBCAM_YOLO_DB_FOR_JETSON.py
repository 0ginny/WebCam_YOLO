import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import Label, Button, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import cx_Oracle
import datetime
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
messagebox.showinfo("device info", f"{device} 로 작동중 입니다.")

# 전역 변수 초기화
model = None
model_names = None
capture_image = None
processed_image = None
save_dir = "./save_img"
model_dir = "./src/yolo_model"
cap = None
last_product_id = None
inspection_started = False  # 검사 시작 여부 추적

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

load_path = './connections/KTLT_AWS.json'
with open(load_path, 'r') as file:
    db_config = json.load(file)

conn = None

def connect_to_db():
    global conn
    try:
        dsn_tns = cx_Oracle.makedsn(db_config["ip"], db_config["port"], db_config["sid"])
        conn = cx_Oracle.connect(db_config["username"], db_config["password"], dsn_tns)
    except cx_Oracle.DatabaseError as e:
        error, = e.args
        messagebox.showerror("Database Error", f"Error Code: {error.code}\nError Message: {error.message}")

connect_to_db()

def color_space_converted_save(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def single_scale_retinex(image, sigma=30):
    def ssr(img, sigma):
        retinex = np.log10(img + 1) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1)
        return retinex

    image = image / 255.0
    retinex_image = ssr(image, sigma)
    retinex_image = (retinex_image - np.min(retinex_image)) / (np.max(retinex_image) - np.min(retinex_image))
    retinex_image = (retinex_image * 255).astype(np.uint8)
    return retinex_image

def apply_preprocessing(image, method):
    if method == 'normal':
        return image
    elif method == 'hsv':
        return color_space_converted_save(image)
    elif method == 'retinex':
        return single_scale_retinex(image)
    else:
        return image

def yolo_obj_draw(draw_img,processed_img, threshold=0.3):
    detection = model(processed_img)[0]
    boxinfos = detection.boxes.data.tolist()

    for data in boxinfos:
        x1, y1, x2, y2 = map(int, data[:4])
        confidence_score = round(data[4], 2)
        classid = int(data[5])
        name = model_names[classid]
        if confidence_score > threshold:
            model_text = f'{name}_{confidence_score}'
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(draw_img, text=model_text, org=(x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(255, 0, 0), thickness=2)

def update_frame():
    global capture_image, preprocessed_image, processed_image
    ret, frame = cap.read()
    if ret:
        capture_image = frame.copy()
        if model is not None:
            processed_image = capture_image.copy()
            preprocessed_image = apply_preprocessing(processed_image, preprocessing_method.get())
            yolo_obj_draw(processed_image, preprocessed_image, threshold=0.2)
            image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        webcam_label.imgtk = image_tk
        webcam_label.configure(image=image_tk)
    webcam_label.after(10, update_frame)

def load_model():
    global model, model_names
    selected_model = model_combobox.get()
    if selected_model and selected_model != "normal":
        model_path = os.path.join(model_dir, selected_model)
        model = YOLO(model_path)
        model.to(device)  # Ensure the model uses GPU
        model_names = model.names
        messagebox.showinfo("Model Loaded", f"{selected_model} 모델을 불러왔습니다")
    else:
        model = None
        model_names = None
        messagebox.showinfo("Model Unloaded", "YOLO 모델이 실행되지 않습니다.")

def generate_product_id(product_code):
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(PRODUCT_ID) FROM PRODUCT_STATE WHERE PRODUCT_ID LIKE :1", (f"{product_code}%",))
    max_id = cursor.fetchone()[0]
    cursor.close()

    if max_id:
        number_part = max_id[len(product_code):]
        new_id_num = int(number_part) + 1
    else:
        new_id_num = 1

    return f"{product_code}{new_id_num:04d}"

def start_webcam():
    global cap
    selected_cam = cam_combobox.get()
    if selected_cam:
        cap_index = int(selected_cam.split(" ")[1])
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
            arr.append(f"Camera {index}")
        cap.release()
        index += 1
    return arr


def start_inspection():
    global product_id_entry, last_product_id, inspection_started
    if inspection_started:
        messagebox.showwarning("Inspection Error", "검사 종료 한 후에 검사를 시작해주세요.")
        return

    product_code = product_code_combobox.get()
    if product_code:
        product_id = generate_product_id(product_code)
        if last_product_id == product_id:
            product_id = generate_product_id(product_code)  # Ensure new product_id if same as last_product_id

        product_id_entry.config(state='normal')
        product_id_entry.delete(0, tk.END)
        product_id_entry.insert(0, product_id)
        product_id_entry.config(state='readonly')

        cursor = conn.cursor()
        cursor.execute("INSERT INTO PRODUCT_STATE (PRODUCT_ID, INSPECTION_START_TIME) VALUES (:1, :2)",
                       (product_id, datetime.datetime.now()))
        conn.commit()
        cursor.close()

        last_product_id = product_id
        inspection_started = True
        messagebox.showinfo("Start Inspection", f"Product ID {product_id} inspection started.")
    else:
        messagebox.showwarning("Input Error", "Please select a Product Code.")

def capture_frame():
    global preprocessed_image
    if preprocessed_image is not None:
        detection = model(preprocessed_image)[0]
        boxinfos = detection.boxes.data.tolist()
        product_id = product_id_entry.get()
        if product_id and boxinfos:
            cursor = conn.cursor()
            for data in boxinfos:
                x1, y1, x2, y2 = map(int, data[:4])
                classid = str(int(data[5])).zfill(2)  # classid를 VARCHAR(2) 형식으로 변환
                width = x2 - x1
                height = y2 - y1
                cursor.execute("INSERT INTO INSPECT_STATE (PRODUCT_ID, ERROR_TYPE, WIDTH, HEIGHT) VALUES (:1, :2, :3, :4)",
                               (product_id, classid, width, height))
            conn.commit()
            cursor.close()
            messagebox.showinfo("Capture Frame", "Captured frame processed and saved.")
        else:
            messagebox.showwarning("Capture Error", "No detection or Product ID is missing.")


def save_image():
    global capture_image, processed_image
    count = len(os.listdir(save_dir)) + 1
    if model is not None and processed_image is not None:
        filename = os.path.join(save_dir, f"capture_{count:03d}.jpg")
        cv2.imwrite(filename, processed_image)
        print(f"Image saved: {filename}")
    elif capture_image is not None:
        filename = os.path.join(save_dir, f"webcam_{count:03d}.jpg")
        cv2.imwrite(filename, capture_image)
        print(f"Image saved: {filename}")
    else:
        messagebox.showwarning("No Image Captured", "캡처된 이미지가 없습니다")


def finish_inspection():
    global last_product_id, inspection_started
    cursor = conn.cursor()
    product_id = product_id_entry.get()
    if product_id:
        cursor.execute(
            "UPDATE PRODUCT_STATE SET INSPECTION_COMPLETE_TIME = :1, IS_DEFECT = (SELECT CASE WHEN EXISTS (SELECT 1 FROM INSPECT_STATE WHERE PRODUCT_ID = :2) THEN 1 ELSE 0 END FROM DUAL) WHERE PRODUCT_ID = :2",
            (datetime.datetime.now(), product_id))
        conn.commit()
        cursor.close()
        messagebox.showinfo("Inspect Finish", f"Product ID {product_id} inspection finished.")

        last_product_id = None  # FINISH 이후 새로 START 시 새로운 PRODUCT_ID 발행을 위해 초기화
        inspection_started = False  # 검사 종료 후 다시 검사 시작 가능하도록 설정
    else:
        messagebox.showwarning("Input Error", "Please enter a Product ID.")

def key_event(event):
    print(f'Key pressed: {event.keysym} (keycode: {event.keycode})')
    if event.keysym in ['Return', 'KP_Enter']:  # Enter and KP_Enter
        print("Enter or KP_Enter pressed")
        capture_frame()
        save_image()
    elif event.keysym in ['KP_Add', 'plus']:  # KP_Add
        print("KP_Add or plus pressed")
        start_inspection()
    elif event.keysym in ['KP_Subtract', 'minus']:  # KP_Subtract
        print("KP_Subtract or minus pressed")
        finish_inspection()

# 모델 파일 리스트 가져오기
model_files = ["normal"] + [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
available_cameras = find_cameras()

product_codes = ["PCBA", "PCBB", "PCBC", "PCBD", "PCBE"]

# GUI 생성
root = tk.Tk()
root.title("YOLO Webcam GUI")
root.bind('<Return>', key_event)
root.bind('<KP_Enter>', key_event)
root.bind('<KP_Add>', key_event)
root.bind('<KP_Subtract>', key_event)
root.bind('<minus>', key_event)
root.bind('<plus>', key_event)

webcam_label = Label(root)
webcam_label.pack(side=tk.LEFT)

capture_label = Label(root)
capture_label.pack(side=tk.LEFT)

init_control_frame = tk.Frame(root)
init_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

Label(init_control_frame, text="사용할 웹캠을 선택한 후 start를 누르세요").grid(row=0, column=0, padx=5, pady=5)
cam_combobox = ttk.Combobox(init_control_frame, values=available_cameras)
cam_combobox.grid(row=1, column=0, padx=5, pady=5)

start_button = Button(init_control_frame, text="Start", command=lambda: [init_control_frame.pack_forget(),
                                                                         control_frame.pack(side=tk.BOTTOM, fill=tk.X,
                                                                                            padx=10, pady=10),
                                                                         start_webcam()])
start_button.grid(row=2, column=0, padx=5, pady=5)

control_frame = tk.Frame(root)

model_combobox = ttk.Combobox(control_frame, values=model_files)
model_combobox.grid(row=0, column=0, padx=5, pady=5)

load_button = Button(control_frame, text="Load Model", command=load_model)
load_button.grid(row=1, column=0, padx=5, pady=5)

# 전처리 방법 선택
Label(control_frame, text="전처리 방법을 선택하세요").grid(row=2, column=0, padx=5, pady=5)
preprocessing_method = ttk.Combobox(control_frame, values=['normal', 'hsv', 'retinex'])
preprocessing_method.grid(row=3, column=0, padx=5, pady=5)
preprocessing_method.current(0)

Label(control_frame, text="Product Code를 선택하세요").grid(row=4, column=0, padx=5, pady=5)
product_code_combobox = ttk.Combobox(control_frame, values=product_codes)
product_code_combobox.grid(row=5, column=0, padx=5, pady=5)

Label(control_frame, text="Product ID").grid(row=6, column=0, padx=5, pady=5)
product_id_entry = tk.Entry(control_frame, state='readonly')
product_id_entry.grid(row=7, column=0, padx=5, pady=5)

Label(control_frame, text="+ 키를 누르면, 검사가 시작됩니다").grid(row=8, column=0, padx=5, pady=5)
Label(control_frame, text="- 키를 누르면, 검사가 종료됩니다").grid(row=9, column=0, padx=5, pady=5)

stop_button = Button(control_frame, text="Stop", command=stop_webcam)
stop_button.grid(row=10, column=0, padx=5, pady=5)

root.mainloop()

# Release the webcam and close windows when the GUI is closed
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
