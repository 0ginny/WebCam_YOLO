import tkinter as tk
from tkinter import messagebox

def button_clicked1():
    messagebox.showinfo("Button1 Clicked", "Button was clicked!")

def button_clicked2():
    messagebox.showinfo("Button2 Clicked", "Button was clicked!")

def button_clicked3():
    messagebox.showinfo("Button3 Clicked", "Button was clicked!")

def key_event(event):
    if event.keysym == '5':
        event.widget.invoke()

root = tk.Tk()
root.title("Button Click Event Example")

# Button 생성
button1 = tk.Button(root, text="Button 1", command=button_clicked1)
button1.pack(pady=10)
button1.bind('<KeyPress-5>', key_event)

button2 = tk.Button(root, text="Button 2", command=button_clicked2)
button2.pack(pady=10)
button2.bind('<KeyPress-5>', key_event)

button3 = tk.Button(root, text="Button 3", command=button_clicked3)
button3.pack(pady=10)
button3.bind('<KeyPress-5>', key_event)

# Focus 설정: Tab 키로 이동 가능
button1.focus_set()

root.mainloop()
