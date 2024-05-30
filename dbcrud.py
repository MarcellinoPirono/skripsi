import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
from tkinter import *
from datetime import datetime
import cv2
import os
from PIL import Image
import numpy as np
from keras_facenet import FaceNet

now = datetime.now()

def GetValue(event):
    e1.delete(0, END)
    e2.delete(0, END)
    e3.delete(0, END)
   #  e4.delete(0, END)
    row_id = listBox.selection()[0]
    select = listBox.set(row_id)
    e1.insert(0,select['IDsiswa'])
    e2.insert(0,select['Nama'])
    e3.insert(0,select['Tanggal'])
   #  e4.insert(0,select['salary'])

def Add():
    idsiswa = e1.get()
    nama = e2.get()
    date_time = e3.get()
    # feee = e4.get()

    mysqldb=mysql.connector.connect(host="127.0.0.1",user="root",password="root",database="siswa")
    mycursor=mysqldb.cursor()

    try:         
      mycursor.execute("SELECT * from datasiswa")
      myresult=mycursor.fetchall()
      id=1
      for x in myresult:
        id+=1
      date_time = now.strftime("%Y-%m-%d %H:%M:%S")
      sql = "INSERT INTO  datasiswa (id,IDsiswa,Nama,Tanggal) VALUES (%s, %s, %s, %s)"
      val = (id,idsiswa,nama,date_time)
      mycursor.execute(sql, val)
      mysqldb.commit()
      face_classifier = cv2.CascadeClassifier("lib/haarcascade_frontalface_default.xml")
      def face_cropped(img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.5,5)
        #scaling factor=1.3
        #Minimum neighbor = 5
 
        if faces is ():
            return None
        for(x,y,w,h) in faces:
            cropped_face=img[y:y+h,x:x+w]
        return cropped_face
 
      cap = cv2.VideoCapture(0)
      img_id=0

      for i in range(5):
        cap.read()
 
      while True:
        ret,frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame),(200,200))
            file_name_path = "dataset/user."+str(id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)
            # (50,50) is the origin point from where text is to be written
            # font scale=1
            #thickness=2
 
            cv2.imshow("Cropped face",face)
            if cv2.waitKey(1000)==13 or int(img_id)==20:
                break
      cap.release()
      cv2.destroyAllWindows()
      messagebox.showinfo('Result','Generating dataset completed!!!')

      lastid = mycursor.lastrowid
      messagebox.showinfo("information", "Siswa berhasil ditambahkan...")
      e1.delete(0, END)
      e2.delete(0, END)
      e3.delete(0, END)
      #e4.delete(0, END)
      listBox.delete(*listBox.get_children())

      show()
    except Exception as e:
      print(e)
      mysqldb.rollback()
      mysqldb.close()


def update():
    id = e1.get()
    Nama = e2.get()
    Tanggal = e3.get()
   #  feee = e4.get()
    mysqldb=mysql.connector.connect(host="127.0.0.1",user="root",password="root",database="siswa")
    mycursor=mysqldb.cursor()

    try:
       sql = "Update datasiswa set Nama= %s,Tanggal= %s where IDsiswa= %s"
       val = (Nama,Tanggal,id)
       mycursor.execute(sql, val)
       mysqldb.commit()
       lastid = mycursor.lastrowid
       messagebox.showinfo("information", "Data berhasil diperbarui...")

       e1.delete(0, END)
       e2.delete(0, END)
       e3.delete(0, END)
      #  e4.delete(0, END)
       e1.focus_set()
       
       listBox.delete(*listBox.get_children())

       show()
       Listbox.update()

    except Exception as e:

       print(e)
       mysqldb.rollback()
       mysqldb.close()

def delete():
    idsiswa = e1.get()

    mysqldb=mysql.connector.connect(host="127.0.0.1",user="root",password="root",database="siswa")
    mycursor=mysqldb.cursor()

    try:
       sql = "delete from datasiswa where IDsiswa = %s"
       val = (idsiswa,)
       mycursor.execute(sql, val)
       mysqldb.commit()
       lastid = mycursor.lastrowid
       messagebox.showinfo("information", "data berhasil dihapus...")

       e1.delete(0, END)
       e2.delete(0, END)
       e3.delete(0, END)
      #  e4.delete(0, END)
       e1.focus_set()
       listBox.delete(*listBox.get_children())

       show()

    except Exception as e:

       print(e)
       mysqldb.rollback()
       mysqldb.close()

def show():
        global records

        mysqldb = mysql.connector.connect(host="127.0.0.1", user="root", password="root", database="siswa")
        mycursor = mysqldb.cursor()
        mycursor.execute("SELECT IDsiswa,nama,tanggal FROM datasiswa")
        records = mycursor.fetchall()
        # print(records)

        global count
        count = 0

        mysqldb.close()

        for i, (IDsiswa,nama, tanggal) in enumerate(records, start=1):
            listBox.insert("", "end", values=(IDsiswa, nama, tanggal))
            mysqldb.close()

def select_record():
    # Clear Box
    e1.delete(0, END)
    e2.delete(0, END)
    e3.delete(0, END)

    # Record Numbers
    selected = listBox.focus()
    # Record Value
    values = listBox.item(selected, 'values')
    
    #output
    e1.insert(0, values[0])
    e2.insert(0, values[1])
    e3.insert(0, values[2])

def preprocess_image(image, target_size):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.resize(target_size)  # Resize the image using PIL
    image_array = np.array(image)  # Convert the PIL image to a numpy array
    
    # Ensure image has 3 channels if it's grayscale or has an alpha channel
    if image_array.ndim == 2:
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] > 3:  # Remove alpha channel if present
        image_array = image_array[:, :, :3]
    
    image_array = image_array.astype('float32')  # Ensure the type is float32
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Since normalization is handled by the library, we skip it here
    return image_array

def get_embedding(model, face_pixels):
    processed_pixels = preprocess_image(face_pixels, (160, 160))
    # Assuming the method to get embeddings is embeddings(), replace with the actual method
    embedding = model.embeddings(processed_pixels)
    return embedding[0]

def train_classifier():
    facenet_model = FaceNet()  # Ensure this correctly loads your FaceNet model
    data_dir = "dataset"
    embeddings = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            id = int(filename.split('.')[1])
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path)  # Open image with PIL
            embedding = get_embedding(facenet_model, img)  # Pass the PIL image directly
            embeddings.append(embedding)
            labels.append(id)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Debugging: Check if all embeddings are identical
    if np.all(embeddings == embeddings[0]):
        print("Warning: All generated embeddings are identical.")
    else:
        print("Embeddings look diverse.")

    # Save embeddings and labels
    np.save('Data/embeddings.npy', embeddings)
    np.save('Data/labels.npy', labels)

    messagebox.showinfo('Result', 'Training dataset completed')

# def refresh():  
#     mysqldb = mysql.connector.connect(host="127.0.0.1", user="root", password="root", database="siswa")
#     listBox.delete(*listBox.get_children())
 
#     mycursor = mysqldb.cursor()
#     display = mycursor.fetchall()

#     for row in display:
#         listBox.insert('','end',value=row)

root = Tk()
root.geometry("800x500")
# root.state("zoomed")
global e1
global e2
global e3
# global e4

tk.Label(root, text="Daftar Siswa", fg="red", font=(None, 24)).place(x=300, y=5)

tk.Label(root, text="ID Siswa").place(x=10, y=10)
Label(root, text="Nama").place(x=10, y=40)
Label(root, text="Tanggal").place(x=10, y=70)
# Label(root, text="Salary").place(x=10, y=100)

e1 = Entry(root)
e1.place(x=140, y=10)

e2 = Entry(root)
e2.place(x=140, y=40)

e3 = Entry(root)
e3.place(x=140, y=70)

# e4 = Entry(root)
# e4.place(x=140, y=100)

Button(root, text="Add",command = Add,height=3, width= 13).place(x=30, y=130)
Button(root, text="Update",command = update,height=3, width= 13).place(x=140, y=130)
Button(root, text="Delete",command = delete,height=3, width= 13).place(x=250, y=130)
Button(root, text="Select",command = select_record,height=3, width= 13).place(x=360, y=130)
Button(root, text="Training",command = train_classifier,height=3, width= 13).place(x=470, y=130)

# # Add Style
# style = ttk.Style()

# #Theme
# style.theme_use('default')

# #Treeeview Colors
# style.configure("Treeview",
#         background="#D3D3D3",
#         foreground="black",
#         rowheight="25",
#         fieldbackground="#D3D3D3")

# # Change Selected Color
# style.map('Treeview',
#         background =[('selected', "#347083")])

# # Treeview Frame
# tree_frame = Frame(root)
# tree_frame.pack(pady=10)
# tree_frame.place(x=40, y=200)

# # Treeview Scrollbar
# tree_scroll = Scrollbar(tree_frame)
# tree_scroll.pack(side=RIGHT, fill=Y)

# # Create the Treeview
# my_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, selectmode="extended")
# my_tree.pack()

# # Configure the Scrollbar
# tree_scroll.config(command=my_tree.yview)

# # Define Columns
# my_tree['columns'] = ("IDSiswa", "Nama", "Tanggal")

# # Format our Column
# my_tree.column("#0", width=0, stretch=NO)
# my_tree.column("IDSiswa", anchor=W, width=140)
# my_tree.column("Nama", anchor=W, width=140)
# my_tree.column("Tanggal", anchor=CENTER, width=140)

# # Create heading
# my_tree.heading("#0", text="", anchor=W)
# my_tree.heading("IDSiswa", text="IDSiswa", anchor=W)
# my_tree.heading("Nama", text="Nama", anchor=W)
# my_tree.heading("Tanggal", text="Tanggal", anchor=CENTER)

# # Create Striped Row Tags
# my_tree.tag_configure('oddrow', background="white")
# my_tree.tag_configure('evenrow', background="lightblue")

# Add data to the screen
# global count
# count = 0

# for record in records:
#     if count % 2 == 0:
#         my_tree.insert(parent='', index='end', iid=count, text='', values=(record[0], record[1], record[2]), tags=('evenrow',))
#     else:
#         my_tree.insert(parent='', index='end', iid=count, text='', values=(record[0], record[1], record[2]), tags=('oddrow',))
#     #increment count
#     count += 1


cols = ('IDSiswa', 'nama', 'tanggal')
listBox = ttk.Treeview(root, columns=cols, show='headings' )

for col in cols:
    listBox.heading(col, text=col)
    listBox.grid(row=1, column=0, columnspan=2)
    listBox.place(x=10, y=200)

scrollbar = ttk.Scrollbar(root, orient='vertical', command=listBox.yview)
scrollbar.place(x=594, y=201, height=224)

# Configure listBox to use the scrollbar
listBox.configure(yscrollcommand=scrollbar.set)

show()
# listBox.bind('<Double-Button-1>',GetValue)

root.mainloop()