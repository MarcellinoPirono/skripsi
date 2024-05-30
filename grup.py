import mysql.connector
from tkinter import ttk, messagebox
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image
import numpy as np
from datetime import datetime

now = datetime.now()

db = mysql.connector.connect(
        host = "localhost",
        user = "root",
        passwd = "root",
        database = "siswa",
    )

cursor = db.cursor()

sql = "SELECT name FROM grup"

name = []
try:
    cursor.execute(sql)
    results = cursor.fetchall()
    for a in results:
        data =  (a[0])
        name.append(data)
        print (data)

except:
    print("Error: unable to fecth data")

def close():
    root.destroy()
    root.update()

root = Tk()
root.title("Group")

root.geometry("250x250")
root.resizable(0, 0)

selected = StringVar(root)
selected.set(name[0])

selected.set("-- Pilih --")
dropdown = OptionMenu(*(root, selected) + tuple(name))
dropdown.place(x=40, y=50)
dropdown.config(width= 12)

Button(root, text="OK",height=1, width= 6).place(x=40, y=100)
Button(root, text="Cancel", command=close, height=1, width= 6).place(x=100, y=100)

label = Label(root,text="Pilih Group",font=("Times 15 bold"))
label.place(x=50, y=5)

root.mainloop()
# db.close()