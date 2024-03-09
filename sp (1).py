import numpy as np
import argparse
import cv2
import tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from keras.models import load_model
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import speech_recognition as sr
import cv2
import threading
#import pymysql as mdb
from pygame import mixer  # Load the popular external library
import time
#import PyAudio
mixer.init()
##import wikipedia
##wikipedia.set_lang("en")
#print (wikipedia.summary("What is ohms law", sentences=5))
mic_name = "Headset Microphone (Realtek(R) "
#Sample rate is how often values are recorded 
sample_rate = 48000
#Chunk is like a buffer. It stores 2048 samples (bytes of data) 
#here.  
#it is advisable to use powers of 2 such as 1024 or 2048 
chunk_size = 2048
#Initialize the recognizer 
r = sr.Recognizer() 
import pymysql as mdb
#generate a list of all audio cards/microphones 
mic_list = sr.Microphone.list_microphone_names() 
print (mic_list)

#the following loop aims to set the device ID of the mic that 
#we specifically want to use to avoid ambiguity. 
for i, microphone_name in enumerate(mic_list): 
    if microphone_name == mic_name: 
        device_id = i
def push(b,c):
            con = mdb.connect('127.0.0.1', \
                              'root', \
                              '', \
                              'hms' );
            print('yes')
            cur = con.cursor()
            #cur.execute("TRUNCATE TABLE `sens`")
            cur.execute("""INSERT INTO tblpatient(PatientAdd,PatientEmail) \
                       VALUES(%s,%s)""", (b,c))
            
            con.commit()

def cam():
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = load_model('emotion_recognition.h5')
    cap = cv2.VideoCapture(0)

    faceCascade = face_detector
    font = cv2.FONT_HERSHEY_SIMPLEX


    emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}





    frame_count = 0


   
    while True:
        ret, frame = cap.read()

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        start_time = time.time()
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        
        y0 = 15
        for index in range(6):
            cv2.putText(frame, emotions[index] + ': ', (5, y0), font,
                        0.4, (255, 0, 255), 1, cv2.LINE_AA)
            y0 += 15

       
        FIRSTFACE = True
        if len(faces) > 0:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height,x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1,48,48,1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                # Probablities of all classes
                #Finding class probability takes approx 0.05 seconds
                start_time = time.time()
                if frame_count % 5 == 0:
                    probab = model.predict(test_image)[0] * 100
                    #print("--- %s seconds ---" % (time.time() - start_time))

                    #Finding label from probabilities
                    #Class having highest probability considered output label
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    frame_count = 0

                frame_count += 1
                font_size = width / 300
                filled_rect_ht = int(height / 5)
                #Drawing probability graph for first detected face
                if FIRSTFACE:
                    y0 = 8
                    for score in probab.astype('int'):
                        cv2.putText(frame, str(score) + '% ', (80 + score, y0 + 8),
                                    font, 0.3, (0, 0, 255),1, cv2.LINE_AA)
                        cv2.rectangle(frame, (75, y0), (75 + score, y0 + 8),
                                      (0, 255, 255), cv2.FILLED)
                        y0 += 15
                        FIRSTFACE =False

                
                #Drawing rectangle and showing output values on frame
                cv2.rectangle(frame, (x, y), (x + width, y + height),(155,155, 0),2)
                cv2.rectangle(frame, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                              (155, 155, 0),cv2.FILLED)
                cv2.putText(frame, predicted_emotion+' '+ str(probab_predicted)+'%',
                            (x, y + height+ filled_rect_ht-10), font,font_size,(255,255,255), 1, cv2.LINE_AA)

               

       

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   print('k')
            
def q1():
    mixer.init()
    mixer.music.load('q1.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                print('ok')
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5
                    
            
def q2():
    mixer.init()
    mixer.music.load('q2.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5
def q3():
    mixer.init()
    mixer.music.load('q3.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return al
               
               
               
                
                
        except:
             a1= 5
def q4():
    mixer.init()
    mixer.music.load('q4.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5
def q5():
    mixer.init()
    mixer.music.load('q5.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5
def q6():
    mixer.init()
    mixer.music.load('q6.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5
def q7():
    mixer.init()
    mixer.music.load('q7.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5
def q8():
    mixer.init()
    mixer.music.load('q8.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5

def q9():
    mixer.init()
    mixer.music.load('q9.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    n=0
    while n==0:
        try:
            with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                                    chunk_size = chunk_size) as source: 
                
                r.adjust_for_ambient_noise(source) 
                print ("Say Ans")
                
                audio = r.listen(source) 
                    
                 
                text = r.recognize_google(audio) 
                print ("you said: " + text )
                text=str(text)            
                #print (wikipedia.summary(text, sentences=5))
                a1=55
                if text=='not at all':
                    a1=0
                    n=1
                    return a1
                elif text=='several days':
                    a1=1
                    n=1
                    return a1
                elif text=='more than half the day':
                    a1=2
                    n=1
                    return a1
                elif text=='nearly every day':
                    a1=3
                    n=1
                    return a1
               
               
               
                
                
        except:
             a1= 5

t1 = threading.Thread(target=cam)
t1.start()
#push(12,'sur5gk@gmail.com')
co=0
while True:
    mydb = mdb.connect(
          host="127.0.0.1",
          user="root",
          passwd="",
          database="hms"
        )

    mycursor = mydb.cursor()

    sql = "SELECT username FROM userlog"


    mycursor.execute(sql)
    fine=0
    myresult = mycursor.fetchall()
    mydb.close()
    for x in myresult:            
        usr=str(x[0])
        print(usr)
    fs=[]
    a= (q1())
    print (a)
    fs.append(a)
    b= (q2())
    print (b)
    fs.append(b)
    c= (q3())
    print (c)
    fs.append(c)
    d= (q4())
    print (d)
    fs.append(d)
    e= (q5())
    print (e)
    fs.append(e)
    f= (q6())
    print (f)
    fs.append(e)
    g= (q7())
    print (g)
    fs.append(g)
    h= (q8())
    print (h)
    fs.append(h)
    print(fs)
    for i in fs: 
        if(i == 3) or (i == 2): 
            co=co+1
    if co>4:
        print ('Major Depressive Disorder')
        res='Major Depressive Disorder'
    elif co>2:
        print ('Depressive Disorder')
        res='Depressive Disorder'
    else:
        print('Consider table')
        for i in fs: 
            i=i+i
        print ('Total weight')
        print (i)
        res=i
    push(res,usr)
        
       
