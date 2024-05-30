import cv2
import threading
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import sys
import time
import base64
import os
import numpy as np
import mysql.connector
from datetime import datetime
import math
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import argparse
import imutils
import dlib
from functools import partial
import customtkinter as ttk
from keras_facenet import FaceNet
from keras.models import load_model
from scipy.spatial.distance import cosine
# import cv2.cuda as cv2_cuda


global count
count =0
global status_up
drowsy = 0

cam_on = False
# cap = None
predictor = dlib.shape_predictor('lib/shape_predictor_68_face_landmarks.dat')
faceCascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

embedder = FaceNet()
embeddings = np.load('data/embeddings.npy')
labels = np.load('data/labels.npy')

# caturing vars
capture_is_busy = False
capture_busy = False
capture_delay = 5
capture_n_second = 0
capture_second = 0
capture_face_detected = False
capture_face_recognized = False


EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 11
YAWN_FRAMES = 11
YAWN_THRESH = 11
alarm_status = False
alarm_status2 = False
saying = False
# COUNTER = 0
recognized_id = None
now = datetime.now()

capture_n_face = 0
# capture_face_img[] = ""       #image base 64

class stopwatch():
    global mydb, mycursor
    mydb=mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="root",
                database="siswa"
                )
    mycursor=mydb.cursor()
    
    def reset(self):
        global count
        count=1
        self.t.set('00:00:00') 
    def start(self):
        global count
        count=0
        self.timer()
        # self.detect()   
    def stop(self):
        global count
        count=1
    def close(self):
        self.root.destroy()


    def timer(self):
        global count
        if(count==0):
            self.d = str(self.t.get())
            h,m,s = map(int,self.d.split(":")) 
            h = int(h)
            m=int(m)
            s= int(s)
            if(s<59):
                s+=1
            elif(s==59):
                s=0
                if(m<59):
                    m+=1
                elif(m==59):
                    m=0
                    h+=1
            if(h<10):
                h = str(0)+str(h)
            else:
                h= str(h)
            if(m<10):
                m = str(0)+str(m)
            else:
                m = str(m)
            if(s<10):
                s=str(0)+str(s)
            else:
                s=str(s)
            self.d=h+":"+m+":"+s           
            self.t.set(self.d)
            if(count==0):
                self.root.after(1000,self.timer)

        if self.drowsy_individuals:
            self.batch_insert_drowsy_individuals(self.drowsy_individuals)
            self.update_result_periodically(self.drowsy_individuals)
            self.drowsy_individuals.clear()

    # def detect(self):
    #     if self.drowsy_individuals:
    #         self.batch_insert_drowsy_individuals(self.drowsy_individuals)
    #         self.update_result_periodically(self.drowsy_individuals)
    #         self.drowsy_individuals.clear()
    #         self.root.after(1000,self.timer)

    def refresh_face_recognizer(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(1, 6, 8, 8)
        self.face_recognizer.read('Kode-1/tes/classifier.xml')

    def detect_face(self, img):
        print("detect face")

        global mycursor, COUNTER, id, drowsy, capture_second, capture_busy, clf, faceCascade, recognize

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = faceCascade.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        
        def final_ear(shape):
            # global final_ear
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            return (ear, leftEye, rightEye)

        def eye_aspect_ratio(eye):
            # global eye_aspect_ratio
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])

            C = dist.euclidean(eye[0], eye[3])

            ear = (A + B) / (2.0 * C)

            return ear
        
        def lip_distance(shape):
            # global lip_distance
            top_lip = shape[50:53]
            top_lip = np.concatenate((top_lip, shape[61:64]))

            low_lip = shape[56:59]
            low_lip = np.concatenate((low_lip, shape[65:68]))

            top_mean = np.mean(top_lip, axis=0)
            low_mean = np.mean(low_lip, axis=0)

            distance = abs(top_mean[1] - low_mean[1])
            return distance


    def enterdb(self):
        # capture_is_busy = False
        # capture_n_second = 0
        global capture_n_second, capture_is_busy
        # print("tes")    
        tes2 = " ".join(tes1)
        # print(type(tes1))
        # id = 1
        # idgroup = 1
        status = "tidur"
        
        nama = "select id from grup where name=%s"
        # nama = "select id from grup where name=" + grups
        val = (tes2,)
        tes22 = mycursor.execute(nama, val)
        records = mycursor.fetchone()[0]
        # records1 = records[1:1]

        # res = int(records)
        # print(records)
        # print(type(records)) 

        # nama = mycursor.fetchone()
        tgl = now.strftime("%Y-%m-%d %H:%M:%S")
        
        capture_n_second = capture_n_second+1
        # print("sebelum ", capture_n_second)
        if(capture_n_second>=capture_delay):
            # print("sesudah ", capture_n_second)
            capture_n_second=0

            if(not capture_is_busy):
                # time.sleep(5)
                capture_is_busy = True
                sql = "INSERT INTO event (tgl, idgroup, IDsiswa, status) VALUES (%s,%s,%s,%s)"
                val = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), records, id, drowsy)
                mycursor.execute(sql, val)
                mydb.commit()
                print("Data committed")
                
                capture_is_busy = False

    def timer_loop(self):
        global capture_busy, capture_second

        capture_second = capture_second+1
        print("captur", capture_second)

        if(capture_second>=5):
            capture_second=0
            print("capturing : ", capture_second)

    def update_result_periodically(self, drowsy_individuals):
        global distance
        drowsy_individuals = list(dict.fromkeys(drowsy_individuals))
        if drowsy_individuals:
            for individual in drowsy_individuals:
                recognized_id, s = individual
                self.insert_result(datetime.now().strftime("%H:%M:%S") + " - " + str(s) + " Mengantuk")

    def preprocess_image(image, target_size, self):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        return image

    def get_embedding(model, face, self):
        face_array = self.preprocess_image(face, (160, 160))

        # Get embedding
        embedding = model.predict(face_array)
        return embedding[0]

    def find_closest_embedding(self, face_embedding, embeddings):
        min_dist = 1.0
        min_id = None
        for i, emb in enumerate(embeddings):
            dist = cosine(face_embedding, emb)  # Ensure both face_embedding and emb are numpy arrays
            if dist < min_dist:
                min_dist = dist
                min_id = labels[i]
        return min_id
    
    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return euler_angle

    def show_frame(self):
        global capture_second, capture_busy, COUNTER, drowsy, s, distance, recognized_id, ear, drowsy_individuals
        # img = None
        ret, frame = self.video_comp.get_frame()

        if cam_on:

            start_time = time.time()
            

            global img, recognized_id, capture_busy, capture_second
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features = faceCascade.detectMultiScale(gray_image, 1.4, 4)
            faces = detector(gray_image, 0)
            recognize_ids=[]
            # drowsiness_state = {}

            for (x, y, w, h) in features:
                face_img = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                face_img_resized = cv2.resize(face_img, (160, 160))
                face_img_expanded = np.expand_dims(face_img_resized, axis=0)

                embedding = embedder.embeddings(face_img_expanded)[0]

                recognized_id = self.find_closest_embedding(embedding, embeddings)

                if recognized_id is not None:
                    mycursor.execute("select Nama from datasiswafn where id=" + str(recognized_id))
                    s = mycursor.fetchone()
                    s = '' + ''.join(s)
                    recognize_ids.append(recognized_id)
                    cv2.putText(frame, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)

                    # Initialize drowsiness state for the recognized ID

                    # Drowsiness detection logic
                    self.perform_drowsiness_check(gray_image, x, y, w, h, recognized_id, frame)
                else:
                    cv2.putText(frame, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = faceCascade.detectMultiScale(gray, scaleFactor=1.1, 
                minNeighbors=8, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            end_time = time.time()
            elapsed_time = end_time - start_time

            if elapsed_time > 0:
                fps = 1 / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if ret:
                
                prevImg = Image.fromarray(frame).resize((950,540))
                # prevImg = Image.fromarray(frame).resize((1000,630))

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=prevImg)
                
                self.vid_lbl.imgtk = imgtk
                self.vid_lbl.configure(image=imgtk)

            self.vid_lbl.after(30, self.show_frame)

    def perform_drowsiness_check(self, gray_image, x, y, w, h, recognized_id, frame):
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray_image, rect)
        shape_np = face_utils.shape_to_np(shape)
        euler_angle = self.get_head_pose(shape_np)

        cv2.putText(frame, f"X: {euler_angle[0, 0]:7.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Y: {euler_angle[1, 0]:7.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Z: {euler_angle[2, 0]:7.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if euler_angle[1, 0] <= 60:  # Assuming the Y-axis pitch angle check
            self.process_drowsiness(shape_np, recognized_id, frame, euler_angle)
        else:
            print("tessssssssss")
            self.drowsiness_state[recognized_id] = {'COUNTER': 0, 'COUNTERS': 0, 'DROWSY': False, 'YAWN': False, 'REPORTED': False}


    def process_drowsiness(self, shape, recognized_id, frame, euler_angle):
        def final_ear(shape):
            # global final_ear
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            return (ear, leftEye, rightEye)

        def eye_aspect_ratio(eye):
            # global eye_aspect_ratio
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])

            C = dist.euclidean(eye[0], eye[3])

            ear = (A + B) / (2.0 * C)

            return ear
        
        def lip_distance(shape):
            # global lip_distance
            top_lip = shape[50:53]
            top_lip = np.concatenate((top_lip, shape[61:64]))

            low_lip = shape[56:59]
            low_lip = np.concatenate((low_lip, shape[65:68]))

            top_mean = np.mean(top_lip, axis=0)
            low_mean = np.mean(low_lip, axis=0)

            distance = abs(top_mean[1] - low_mean[1])
            return distance
                
        if recognized_id not in self.drowsiness_state:
            self.drowsiness_state[recognized_id] = {'COUNTER': 0, 'COUNTERS': 0, 'DROWSY': False, 'YAWN': False}

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Debugging: Print EAR and Yawn Distance values
        # print(f"ID: {id}, EAR: {ear:.2f}, Yawn Distance: {distance}")

        if ear < EYE_AR_THRESH:
            self.drowsiness_state[recognized_id]['COUNTER'] += 1
            # print(f"ID: {id}, EAR: {ear:.2f}, COUNTER: {drowsiness_state[id]['COUNTER']}")  # Debug print

            if self.drowsiness_state[recognized_id]['COUNTER'] >= EYE_AR_CONSEC_FRAMES:
                self.drowsiness_state[recognized_id]['DROWSY'] = True
                self.drowsy_individuals.append((recognized_id, s))
                self.drowsiness_state[recognized_id]['REPORTED'] = True

        else:
            self.drowsiness_state[recognized_id]['COUNTER'] = 0
            self.drowsiness_state[recognized_id]['DROWSY'] = False
            self.drowsiness_state[recognized_id]['REPORTED'] = False
            # print(f"ID: {id}, EAR: {ear:.2f}, Not Drowsy")  # Debug print

        if distance > YAWN_THRESH:
            self.drowsiness_state[recognized_id]['COUNTERS'] += 1

            if self.drowsiness_state[recognized_id]['COUNTERS'] >= YAWN_FRAMES:
                self.drowsiness_state[recognized_id]['YAWN'] = True
                self.drowsy_individuals.append((recognized_id, s))
                self.drowsiness_state[recognized_id]['REPORTED'] = True
        else:
            self.drowsiness_state[recognized_id]['COUNTERS'] = 0
            self.drowsiness_state[recognized_id]['YAWN'] = False
            self.drowsiness_state[recognized_id]['REPORTED'] = False

        #debugging information
        print(f"ID: {recognized_id}, EAR: {ear:.2f}, COUNTER: {self.drowsiness_state[recognized_id]['COUNTER']}")

        # Display the drowsiness state for each recognized ID
        drowsy_status_text = "Drowsy" if self.drowsiness_state[recognized_id]['DROWSY'] else "Alert"
        yawn_status_text = "Yawning" if self.drowsiness_state[recognized_id]['YAWN'] else "Not Yawning"
        print(f"ID: {recognized_id}, yawn: {distance:.2f}")
        # cv2.putText(frame, f"ID {recognized_id}: {drowsy_status_text}, {yawn_status_text}", (x, y - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)

    def batch_insert_drowsy_individuals(self, drowsy_individuals):
        global capture_detik, capture_sibuk

        drowsy_individuals = list(dict.fromkeys(drowsy_individuals))
        print (drowsy_individuals)
        tes2 = " ".join(tes1)
        nama = "select id from grup where name=%s"
        # nama = "select id from grup where name=" + grups
        val = (tes2,)
        tes22 = mycursor.execute(nama, val)
        records = mycursor.fetchone()[0]

        if drowsy_individuals:
            query = "INSERT INTO event (tgl, idgroup, IDsiswa, status) VALUES (%s, %s, %s, %s)"
            values = []

            for individual in drowsy_individuals:
                recognized_id, s = individual
                recognized_id_native = int(recognized_id)
                tgl = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                values.append((tgl, records, recognized_id_native, "Drowsy"))

            mycursor.executemany(query, values)
            mydb.commit()
            drowsy_individuals.clear()
            # self.processed_ids.clear()
            print("Batch data committed")
            # capture_sibuk = False    

    def status_update(self):
        global status_up


    def start_vid(self):
        global cam_on, cap, fps, capture_n_second, capture_delay
        # stop_vid()
        print("tesssss")
        cam_on = True
        cap = cv2.VideoCapture(0)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.show_frame()

    def stop_vid(self):
        global cam_on
        cam_on = False
        
        if cap:
            cap.release()

    def database(self):
        os.system('python dbcrud.py')
        # self.root = tk.Toplevel()

    def group(self):
        os.system('python grup.py')
        self.root = tk.Toplevel()

    def grup(self):
        global selected, value
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

        root = ttk.CTk()
        root.title("Group")

        root.geometry('350x350+20+570')
        # root.resizable(0, 0)

        selected = StringVar(root)
        # selected.set(name[0])
        results_for_combobox = [result[0] for result in results]

        def change_handler(value):
            global tes1
            print(f"Selected Value {value}")
            tes1 = {value}


        dropdown = ttk.CTkComboBox(root, width=80, values=results_for_combobox, state="readonly", font=ttk.CTkFont(family="Times", size=20), command=change_handler)
        # dropdown = ttk.CTkComboBox(*(root, selected) + tuple(name))
        dropdown.set("-- Pilih --")
        dropdown.place(x=50, y=50)
        dropdown.configure(width= 160, height=30)
        
        def tes(selected):
            global grups
            grups = selected.get()
            print(grups)
            return grups

        ttk.CTkButton(root, text="OK", fg_color='green', corner_radius=10, 
                      hover_color="darkgreen", command=lambda: [self.start_vid(), 
                      self.start(), close()], 
               height=35, width= 70, font=ttk.CTkFont(family="Times Bold", size=16)).place(x=50, y=120)
        ttk.CTkButton(root, text="Cancel", fg_color='red', hover_color="darkred", 
                      corner_radius=10, command=close, height=35, width= 70, 
                      font=ttk.CTkFont(family="Times Bold", size=16)).place(x=130, y=120)
        # Button(root, text="coba", command= partial(tes, selected)).place(x=130, y=130)

        label = ttk.CTkLabel(root,text="Pilih Group", fg_color="transparent", font=ttk.CTkFont(family="Times bold", size=15))
        label.place(x=80, y=5)
        root.mainloop()


    def insert(self):
        while True:
            self.text_box.insert('end', 'hi\n')
            # self.after(1000, self.insert)
            time.sleep(2)

    def insert_text(self):
        x = threading.Thread(target=self.insert)
        x.daemon = True
        x.start()

    def insert_result(self, result):
        self.text_box.config(state=tk.NORMAL)
        self.text_box.insert(tk.END,"\n"+result)
        self.text_box.config(state=tk.DISABLED) # lock back the textbox to readonly
        self.text_box.see(tk.END)

    def __init__(self):
        global distance
        # if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        #     print("CUDA is available. Using GPU for face detection.")
        # else:
        #     print("CUDA is not available. Using CPU.")
        self.root = ttk.CTk()
        # self.root = self.root()
        self.root.title("frame 1")

        # faceCascade=cv2.CascadeClassifier("Kode-1/tes/haarcascade_frontalface_default.xml")
        self.clf = cv2.face.LBPHFaceRecognizer_create()
        # self.clf.read("Kode-1/tes/classifier/classifier.yml")
        # self.clf.read("Kode-1/tes/classifier.xml")
        distance = 0
        self.processed_ids = set()
        self.drowsiness_state = {}
        self.head_poses = {}
        self.drowsy_individuals = []
        # self.recognize = None  # Initialize recognize as None

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=2)
        self.root.grid_rowconfigure(2, weight=1)

        # open video source (by default this will try to open the computer webcam)
        self.video_comp = VideoCaptureComp()

        width= self.root.winfo_screenwidth() 
        height= self.root.winfo_screenheight()
        #setting tkinter window size
        self.root.geometry("%dx%d+0+0" % (width, height))

        # self.root.geometry("1280x720")
        # self.root.attributes("-fullscreen", "True")
        ttk.set_appearance_mode("dark")
        self.t = StringVar()
        self.t.set("00:00:00")
        self.list = ("Status")

        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                [1.330353, 7.122144, 6.903745],
                                [-1.330353, 7.122144, 6.903745],
                                [-6.825897, 6.760612, 4.402142],
                                [5.311432, 5.485328, 3.987654],
                                [1.789930, 5.393625, 4.413414],
                                [-1.789930, 5.393625, 4.413414],
                                [-5.311432, 5.485328, 3.987654],
                                [2.005628, 1.409845, 6.165652],
                                [-2.005628, 1.409845, 6.165652],
                                [2.774015, -2.080775, 5.048531],
                                [-2.774015, -2.080775, 5.048531],
                                [0.000000, -3.116408, 6.097667],
                                [0.000000, -7.415691, 4.070434]])

        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                [10.0, 10.0, -10.0],
                                [10.0, -10.0, -10.0],
                                [10.0, -10.0, 10.0],
                                [-10.0, 10.0, 10.0],
                                [-10.0, 10.0, -10.0],
                                [-10.0, -10.0, -10.0],
                                [-10.0, -10.0, 10.0]])

        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        
        self.header = ttk.CTkFrame(self.root, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="nswe")
        self.header.grid_columnconfigure(1, weight=0)
        self.daftar = ttk.CTkButton(self.header, text="Daftar Siswa", text_color="#c0c0c0", width= 220,
                                    font=ttk.CTkFont(family="Times", size=30), command=self.database, 
                                    fg_color="grey", hover_color="#525252")
        self.daftar.grid(row=0, column=0, pady=(30, 10), padx=(20, 0), sticky="e")

        self.hitung = ttk.CTkFrame(self.root, fg_color="transparent")
        self.hitung.grid(row=0, column=1, columnspan=2, sticky="nsew")
        # self.header.grid(row=0, column=0, sticky="nsew")
        self.lb = ttk.CTkLabel(self.hitung,textvariable=self.t, 
                               font=ttk.CTkFont(family="Times", size=50), 
                               fg_color="transparent")
        self.lb.grid(row=0, column=0, padx=20, pady=10, sticky="")

        self.deteksi = ttk.CTkFrame(self.root, fg_color="transparent", width=20)
        self.deteksi.grid(row=1, column=0, padx=(20,0), pady=(20, 10), sticky="nsew")
        self.vid_lbl = Label(self.deteksi)
        self.vid_lbl.grid(row=0, column=0, padx=(20,0), pady=(20, 10), sticky="e")

        self.kosong = ttk.CTkFrame(self.root, fg_color="transparent")
        self.kosong.grid(row=1, column=1, padx=(20,0), pady=(20, 10), sticky="nsew") 

        self.siswa = ttk.CTkFrame(self.root, fg_color="transparent")
        self.siswa.grid(row=1, column=2, padx=(20,20), pady=(20, 10), sticky="nsew")
        # self.judul = ttk.CTkLabel(self.siswa, text="UI Scaling:", anchor="w")
        # self.judul.grid(row=0, column=0, padx=20, pady=(20, 10))
        framesiswa = ttk.CTkFrame(self.siswa)
        self.text_box = Text(framesiswa, height=20, width=40, wrap='word')
        # self.text_box.insert(tk.END, "Drowsiness Detection Result: " + result + "\n")

        self.text_box.pack(side=LEFT, expand=True)
        sb = ttk.CTkScrollbar(framesiswa)
        sb.pack(side=RIGHT, fill=BOTH)
        self.text_box.configure(yscrollcommand=sb.set)
        sb.configure(command=self.text_box.yview)
        framesiswa.pack(expand=True)
        framesiswa.grid(row=1, column=0, padx=20, pady=(20, 10))
        font_text = ("Comic Sans MS", 16, "bold")
        self.text_box.configure(font= font_text, state="normal")

        self.siswa = ttk.CTkFrame(self.root, fg_color="transparent")
        self.siswa.grid(row=2, column=0, sticky="nsew")
        self.bt1 = ttk.CTkButton(self.siswa,text="Start", command=self.grup, 
                                 fg_color="green", hover_color="darkgreen", 
                                 width=90, font=ttk.CTkFont(family="Times", size=30))
        self.bt1.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.bt2 = ttk.CTkButton(self.siswa,text="Stop",command=lambda: [self.stop(), self.stop_vid()], 
                                 fg_color="red", hover_color="darkred", 
                                 width=90, font=ttk.CTkFont(family="Times", size=30))
        self.bt2.grid(row=0, column=1, padx=20, pady=(20, 10))
        self.bt3 = ttk.CTkButton(self.siswa,text="Reset",command=self.reset, 
                                 fg_color="orange", hover_color="#944b07", 
                                 width=90, font=ttk.CTkFont(family="Times", size=30))
        self.bt3.grid(row=0, column=2, padx=20, pady=(20, 10))
        
        self.berhenti = ttk.CTkFrame(self.root, fg_color="transparent")
        self.berhenti.grid(row=2, column=2, sticky="nsew")
        self.bt4 = ttk.CTkButton(self.berhenti, text="Exit", text_color='red', command=self.close, 
                                 fg_color="yellow", hover_color="#adad31", 
                                 width=90, font=ttk.CTkFont(family="Times", size=30))
        self.bt4.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.root.mainloop()

class VideoCaptureComp:
    def __init__(self):
        # Open the video Source
        if type(1) is int:
            # video_path = '0'
            self.vid = cv2.VideoCapture(0)
        else:
            # video_path = ''
            self.vid = cv2.VideoCapture(0)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", 1)
        
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self. vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return False, None
        
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()

a=stopwatch()