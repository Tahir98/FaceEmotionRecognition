# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:29:50 2022

@author: Tahir
"""

import cv2
import numpy as np
import tensorflow as tf

new_model = tf.keras.models.load_model("my_model_64p252.h5")

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

rectangle_color = (0,0,255)
img =  np.zeros((1280,720))
text = "Empty text"

(text_width,text_height) = cv2.getTextSize(text,font,font_scale,thickness = 1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25

box_coords = ((text_offset_x,text_offset_y),(text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_color,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,font_scale,color = (0,0,0),thickness = 1)

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    raise IOError("Camera cannot open")

while True:
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
        facess = faceCascade.detectMultiScale(roi_gray)
        
        if len(facess) == 0:
            print("Face not detected")
        else:
            for ex,ey,ew,eh in facess:
                face_roi = roi_color[ey:ey+eh,ex:ex+ew]
                
                final_image = cv2.resize(face_roi,(224,224))
                final_image = np.expand_dims(final_image,axis = 0) #fourth dimension
                final_image = final_image/255.0 #normalization
                
                Predictions = new_model.predict(final_image)
                
                font_scale = 1.5
                font = cv2.FONT_HERSHEY_PLAIN
                
                if np.argmax(Predictions) == 0:
                    status = "Angry"
                    
                    x1,y1,w1,h1 = 0,0,175,175
                    
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                    
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                    
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
                elif np.argmax(Predictions) == 1:
                    status = "Disgust"
                    
                    x1,y1,w1,h1 = 0,0,175,175
                    
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                    
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                    
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
                elif np.argmax(Predictions) == 1:
                    status = "Disgust"
                     
                    x1,y1,w1,h1 = 0,0,175,175
                     
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                     
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                     
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))   
                elif np.argmax(Predictions) == 2:
                    status = "Fear"
                     
                    x1,y1,w1,h1 = 0,0,175,175
                     
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                     
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                     
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))               
                elif np.argmax(Predictions) == 3:
                    status = "Happy"
                     
                    x1,y1,w1,h1 = 0,0,175,175
                     
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                     
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                     
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))            
                elif np.argmax(Predictions) == 4:
                    status = "Sad"
                     
                    x1,y1,w1,h1 = 0,0,175,175
                     
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                     
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                     
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))               
                elif np.argmax(Predictions) == 5:
                    status = "Suprise"
                     
                    x1,y1,w1,h1 = 0,0,175,175
                     
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                     
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                     
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))   
                else:
                    status = "Neutral"
                     
                    x1,y1,w1,h1 = 0,0,175,175
                     
                    cv2.rectangle(frame,(x1,y1),(x1 + w1,y1 + h1),(0,0,0),-1)
                     
                    cv2.putText(frame,status,(x1 + int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                     
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))   
                
    
    cv2.imshow("window1", frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break    
cap.release()   
cv2.destroyAllWindows()    
    
    
    
    
    
    
    
    
    