# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:29:23 2022

@author: Tahir
"""

import cv2
import matplotlib.pyplot as plt 
from deepface import DeepFace
import numpy as np


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                    'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read() 

    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(1200,900))
    
    result = DeepFace.analyze(frame,actions = ['emotion'],enforce_detection = False)

    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)
    font = cv2.FONT_HERSHEY_SIMPLEX


    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        
    cv2.putText(frame,result['dominant_emotion'],(50,100),font,3,(0,255,0),2,cv2.LINE_4)
  
    cv2.imshow("Window1",frame)


    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()