# -*- coding: utf-8 -*-
"""
Created on Wed May 11 00:16:34 2022

@author: Tahir
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import random

new_model = tf.keras.models.load_model("my_model_64p25.h5")

frame = cv2.imread("happyboy.jpg")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,1.1,4)

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

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
            
            if np.argmax(Predictions) == 1:
                status = "Happy"
                
                cv2.putText(frame, status,(100,150),font,3,(0,0,255),cv2.LINE_4)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
            
cv2.imshow("Window1",frame)










