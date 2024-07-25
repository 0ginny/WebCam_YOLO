import cv2
import os
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("./src/240724_yolo_crop_segmentation_pattern.pt")
model_names = model.names

# 캡쳐보드 인식
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture_image = None
processed_image = None
save_dir = "./save_img"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        webcam_label.imgtk = image_tk
        webcam_label.configure(image=image_tk)
        webcam_label.after(10, update_frame)
        capture_image = frame.copy()


def capture():
    global capture_image, capture_label, processed_image
    if capture_image is not None:
        processed_image = capture_image.copy()
        yolo_obj_draw(processed_image, threshold=0.2)
        yolo_img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        yolo_img_pil = Image.fromarray(yolo_img_rgb)
        yolo_img_tk = ImageTk.PhotoImage(yolo_img_pil)
        capture_label.imgtk = yolo_img_tk
        capture_label.configure(image=yolo_img_tk)


def save_image():
    global processed_image
    if processed_image is not None:
        count = len(os.listdir(save_dir)) + 1
        filename = os.path.join(save_dir, f"capture_{count:03d}.jpg")
        cv2.imwrite(filename, processed_image)
        print(f"Image saved: {filename}")


# GUI 생성
root = tk.Tk()
root.title("YOLO Webcam GUI")

webcam_label = Label(root)
webcam_label.pack(side=tk.LEFT)

capture_label = Label(root)
capture_label.pack(side=tk.LEFT)

button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

capture_button = Button(button_frame, text="Capture", command=capture)
capture_button.pack(side=tk.LEFT, padx=5)

save_button = Button(button_frame, text="Save", command=save_image)
save_button.pack(side=tk.LEFT, padx=5)

update_frame()
root.mainloop()

# Release the webcam and close windows when the GUI is closed
cap.release()
cv2.destroyAllWindows()
