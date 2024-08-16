import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import Label, Button, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import mysql.connector
import datetime
import json
from datetime import datetime
import time

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
        conn = mysql.connector.connect(
            host=db_config["ip"],
            user=db_config["username"],
            password=db_config["password"],
            database=db_config["sid"],
            port=db_config["port"]
        )
    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Error Code: {e.errno}\nError Message: {e.msg}")

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

def yolo_obj_draw(draw_img, processed_img, threshold=0.3):
    detection = model(processed_img)[0]
    boxinfos = detection.boxes.data.tolist()

    for data in boxinfos:
        x1, y1, x2, y2 = map(int, data[:4])
        confidence_score = round(data[4], 2)
        classid = int(data[5])
        name = model_names[classid]
        if confidence_score > threshold:
            model_text = f'{name} W:{x2-x1} H:{y2-y1}'
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(draw_img, text=model_text, org=(x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(255, 0, 0), thickness=2)

def update_frame():
    global capture_image, preprocessed_image, processed_image
    global image_tk  # 전역 변수로 선언
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
        image_tk = ImageTk.PhotoImage(image)  # 전역 변수로 설정
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
    cursor.execute("SELECT MAX(PRODUCT_ID) FROM PRODUCT_STATE WHERE PRODUCT_ID LIKE %s", (f"{product_code}%",))
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

    product_code = product_code_combobox.get()  # PCB_TYPE으로 사용
    if product_code:
        # `PCB_PRODUCT` 테이블에 존재하는 `PCB_TYPE` 값인지 확인
        cursor = conn.cursor()
        cursor.execute("SELECT PCB_TYPE FROM PCB_PRODUCT WHERE PCB_TYPE = %s", (product_code,))
        valid_product = cursor.fetchone()

        if not valid_product:
            messagebox.showwarning("Input Error", "선택한 Product Code가 PCB_PRODUCT 테이블에 존재하지 않습니다.")
            cursor.close()
            return

        product_id = generate_product_id(product_code)
        if last_product_id == product_id:
            product_id = generate_product_id(product_code)  # Ensure new product_id if same as last_product_id

        product_id_entry.config(state='normal')
        product_id_entry.delete(0, tk.END)
        product_id_entry.insert(0, product_id)
        product_id_entry.config(state='readonly')

        # `PRODUCT_STATE` 테이블에 삽입
        cursor.execute("INSERT INTO PRODUCT_STATE (PRODUCT_ID, PCB_TYPE, INSPECTION_START_TIME) VALUES (%s, %s, %s)",
                       (product_id, product_code, datetime.datetime.now()))
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
                cursor.execute("INSERT INTO INSPECT_STATE (PRODUCT_ID, ERROR_TYPE, WIDTH, HEIGHT) VALUES (%s, %s, %s, %s)",
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
            "UPDATE PRODUCT_STATE SET INSPECTION_COMPLETE_TIME = %s, IS_DEFECT = (SELECT CASE WHEN EXISTS (SELECT 1 FROM INSPECT_STATE WHERE PRODUCT_ID = %s) THEN 1 ELSE 0 END) WHERE PRODUCT_ID = %s",
            (datetime.datetime.now(), product_id, product_id))
        conn.commit()
        cursor.close()
        messagebox.showinfo("Inspect Finish", f"Product ID {product_id} inspection finished.")

        last_product_id = None  # FINISH 이후 새로 START 시 새로운 PRODUCT_ID 발행을 위해 초기화
        inspection_started = False  # 검사 종료 후 다시 검사 시작 가능하도록 설정
    else:
        messagebox.showwarning("Input Error", "Please enter a Product ID.")

def key_event(event):
    print(f'Key pressed: {event.keysym} (keycode: {event.keycode})')
    if event.keysym in ['asterisk']:  # asterisk for KP_Multiply
        print("Asterisk (*) pressed (Enter functionality)")
        capture_frame()
        save_image()
    elif event.keysym in ['plus']:  # plus
        print("Plus (+) pressed")
        start_inspection()
    elif event.keysym in ['minus']:  # minus
        print("Minus (-) pressed")
        finish_inspection()
    elif event.keysym == '2':  # 아래로 이동
        if isinstance(event.widget, ttk.Combobox):
            # 다음 항목으로 이동
            current_index = event.widget.current()
            if current_index < len(event.widget['values']) - 1:
                event.widget.current(current_index + 1)
    elif event.keysym == '8':  # 위로 이동
        if isinstance(event.widget, ttk.Combobox):
            # 이전 항목으로 이동
            current_index = event.widget.current()
            if current_index > 0:
                event.widget.current(current_index - 1)
    elif event.keysym == '5':
        event.widget.invoke()  # Simulate button click
    elif event.keysym == '7':
        # 포커스를 다음 위젯으로 이동 (탭과 동일한 기능)
        next_widget = event.widget.tk_focusNext()
        if next_widget:
            next_widget.focus_set()

def get_list_from_db(table_name:str, column_name:str, colunm_index:int)->list:
    '''
    db에서 칼럼을 리스트로 반환

    :param table_name: db 테이블명 (str)
    :param column_name: 목표 칼럼명 (str)
    :param colunm_index: 테이블에서 목표 칼럼 인덱스(0~)
    :param target_list: 변환할 list
    :return: list
    '''
    target_list = []
    try:
        # 커서 생성
        cursor = conn.cursor()
        # SQL 쿼리 실행
        cursor.execute(f"SELECT {column_name} FROM {table_name}")
        # 결과 가져오기
        result = cursor.fetchall()
        # 리스트 초기화 및 생성
        target_list = [row[colunm_index] for row in result]

    except mysql.connector.Error as e:
        print(f"{table_name} 테이블에 에러가 발생했습니다: {e}")

    finally:
        # 커서와 연결 종료
        cursor.close()
    return target_list

# 모델 파일 리스트 가져오기
model_files = ["normal"] + [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
available_cameras = find_cameras()

product_codes = get_list_from_db("PCB_PRODUCT", "PCB_TYPE", 0)
print(product_codes)
# GUI 생성
root = tk.Tk()
root.title("YOLO Webcam GUI")

# 전역 키 이벤트 바인딩
root.bind('<asterisk>', key_event)  # Bind asterisk for Enter functionality
root.bind('<plus>', key_event)  # Bind plus for KP_Add
root.bind('<minus>', key_event)  # Bind minus for KP_Subtract
root.bind('<KeyPress-5>', key_event)  # Bind number 5
root.bind('<KeyPress-2>', key_event)  # Bind number 2
root.bind('<KeyPress-8>', key_event)  # Bind number 8
root.bind('<KeyPress-7>', key_event)  # Bind number 7

webcam_label = Label(root)
webcam_label.pack(side=tk.LEFT)

capture_label = Label(root)
capture_label.pack(side=tk.LEFT)

init_control_frame = tk.Frame(root)
init_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

Label(init_control_frame, text="사용할 웹캠을 선택한 후 start를 누르세요").grid(row=0, column=0, padx=5, pady=5)
cam_combobox = ttk.Combobox(init_control_frame, values=available_cameras, state='readonly')
cam_combobox.grid(row=1, column=0, padx=5, pady=5)

start_button = Button(init_control_frame, text="Start", command=lambda: [init_control_frame.pack_forget(),
                                                                         control_frame.pack(side=tk.BOTTOM, fill=tk.X,
                                                                                            padx=10, pady=10),
                                                                         start_webcam()])
start_button.grid(row=2, column=0, padx=5, pady=5)

control_frame = tk.Frame(root)

# 상태를 'readonly'로 설정하여 콤보박스의 직접적인 텍스트 입력을 막음
model_combobox = ttk.Combobox(control_frame, values=model_files, state='readonly')
model_combobox.grid(row=0, column=0, padx=5, pady=5)

load_button = Button(control_frame, text="Load Model", command=load_model)
load_button.grid(row=1, column=0, padx=5, pady=5)

# 전처리 방법 선택
Label(control_frame, text="전처리 방법을 선택하세요").grid(row=2, column=0, padx=5, pady=5)
preprocessing_method = ttk.Combobox(control_frame, values=['normal', 'hsv', 'retinex'], state='readonly')
preprocessing_method.grid(row=3, column=0, padx=5, pady=5)
preprocessing_method.current(0)

Label(control_frame, text="Product Code를 선택하세요").grid(row=4, column=0, padx=5, pady=5)
product_code_combobox = ttk.Combobox(control_frame, values=product_codes, state='readonly')
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
