# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:23:05 2022

@author: Tahir
"""

import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import random

Datadirectory = "train/"
Classes = ["0","1","2","3","4","5","6"]

training_data = []
img_size = 224

def create_training_data():
    for category in Classes:
        cat = category + "/"
        path = os.path.join(Datadirectory, cat)
        class_num = Classes.index(category)
        print(path)
        print(len(os.listdir(path)))
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
 
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
                
                if category == "1" :
                    new_array.flip()
                    training_data.append([new_array,class_num])
                
                    new_array.rotate()
                    training_data.append([new_array,class_num])
                    
                    new_array.rotate()
                    training_data.append([new_array,class_num])
            except Exception as e:
                pass
    
create_training_data()
print(len(training_data))

random.shuffle(training_data)

x = []
y = []

for features,label in training_data:
    x.append(features)
    y.append(label)
    
X = np.array(x).reshape(-1,img_size,img_size,3)    
X = X/np.float32(255.0)
    
print(X.shape)    
    
Y = np.array(y)   

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  
    
model = tf.keras.applications.MobileNetV3Small()

#print(model.summary())
 
base_input = model.layers[0].input    
base_output = model.layers[-2].output    
 
#adding new layer, after the output of global pooling layer
final_output = layers.Dense(128)(base_output)

#activation function
final_ouput = layers.Activation('relu')(final_output)

final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)

#my classes are 7
final_output = layers.Dense(7,activation = 'softmax')(final_ouput)    
    
#new model
new_model = keras.Model(inputs = base_input,outputs = final_output)    
#print(new_model.summary())    

new_model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])
new_model.fit(X,Y,epochs = 25) 
new_model.save('my_model_64p252.h5')
   

