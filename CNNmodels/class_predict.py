# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 22:22:03 2021

@author: WAEL
"""

import PIL.Image as Image
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
#from keras.preprocessing import image
import pathlib
from sklearn.metrics import classification_report
print("for labeling y_test>>>>>>>>>>>>>>>>>>>>>>>>>>>")
DATADIR = r'D:\google test\class_predict'  
CATEGORIES = ["covid", "normal"]
IMG_SIZE = 500
training_data =[]

def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) 
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                #plt.imshow(new_array)
                #plt.show()
            except Exception as e:
                print(e)
                pass
               
            
create_testing_data()

print(len(training_data))

for sample in training_data[:10]:
    print(sample[1])
X_test = []
y_test = []

for features,label in training_data:
     X_test.append(features)
     y_test.append(label)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("it could be named y_true:")
print(y_test)

print(X_test)

print("for predicting y>>>>>>>>>>>>>>>>>>>>>>>")
CATEGORIES =["covid","normal"]
image_height=1500
image_width=1500
def class_prdict(data_dir):
    
    model = tf.keras.models.load_model("32x2x0CNN.model") # relative directory
    def prepare(filepath):
        IMG_SIZE = 500
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
       
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = new_array / 255.0
        #print(new_array)
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
        
        
    
   
    
     #Absolute directory
    data_dir = pathlib.Path(data_dir)
    print(data_dir)
    class_names = np.array([item.name for item in data_dir.glob('*')])# can we take formats of images  ,,* which means all files and directiories ,we can also search for py files>>> *.py or everything *.*
    print(class_names) # right names for here its correct
    print("correct class name")
    #prediction = model.predict([prepare('no.jfif')])
    class_names_arr = class_names.tolist()  # to array?
    print(class_names)  
    pred_list=[]
    for x in range(len(class_names_arr)) :
       
        prediction = model.predict([prepare(class_names_arr[x])])
        pred_list.append(prediction)    
    print(pred_list)
    print("__________________")
    print(pred_list[0][0])
    print(pred_list[1][0])
    print(pred_list[2][0])
    print(pred_list[3][0])
    print(range(len(pred_list)))
    
    
    print(pred_list)
    print("gather them into an array")
    pred_list=np.array(pred_list)
    y_pred=pred_list[:,0].copy()
    print(y_pred)
    print("__________________")
    
    
    print(range(y_pred.shape[0]))
    for i in range(y_pred.shape[0]):
        if y_pred[i]>0.5:
           y_pred[i]=1
        else:
            y_pred[i]=0
    return y_pred

y_predNormal=class_prdict(r'D:/google test/class_predict/normal').tolist()
y_predCovid=class_prdict(r'D:/google test/class_predict/covid').tolist()
print(y_predNormal)
print(y_predCovid)
print("Final step>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
for i in y_predNormal :
    y_predCovid.append(i)

pred_all=y_predCovid.copy()
print(pred_all)
target_names = ['class 0', 'class 1']
print(classification_report(pred_all, y_test, target_names=target_names))
