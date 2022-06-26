#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:11:57 2022

@author: pratyushvivek
"""
SIZE=300
import matplotlib.pyplot as plt
import cv2
import joblib
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,array_to_img,load_img
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn
from sklearn import preprocessing
import firebase_admin
from firebase_admin import credentials,storage


le = preprocessing.LabelEncoder()
# loaded_rf = joblib.load("./random_forest.joblib")
loaded_xgboost=joblib.load("./XGBoost.joblib")
# loaded_svm=joblib.load("./Support_Vector.joblib")
model=load_model('Estrus_model.h5',compile=True)
def Prediction(Image_Path):
    img_path=Image_Path
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #x1=img_to_array(img)
    #n=38#Select the index of image to be loaded for testing
    #img = x_test[n]
    plt.imshow(img)
    input_img = np.expand_dims(img, axis=0)
    #input_img = np.expand_dims(img, axis=0) 
    
    input_img_feature=model.predict(input_img)
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    #Y1=model.predict(x1)
    
    X2=loaded_xgboost.predict(input_img_features)[0]
    
#X1= le.inverse_transform([X1])
    
    
    if X2==0:
            print('Atypical')
            
            
    else:
            print("Typical")
            
            
    print("prediction svm")
    return X2


   
def start():
        
    
#     Image_Path=input("Enter the path of the image for which you want the prediction:    ")
#     prediction=Prediction(Image_Path)
    prediction=Prediction("image.jpg")
    Result=""
    if prediction==0:
        Result="Atypical"
    else:
        Result="Typical"
    print('prediction=',Result)
    return Result

# start()

#prediction2=Prediction(Image_Path)'''
'''import tensorflow as tf

# Convert the model
pipe_clf_params = {}
filename = 'E:/strokestuff/strokelrpred/strokelrpred.joblib'

pipe_clf_params['pipeline'] = lrmodel
joblib.dump(pipe_clf_params, filename)
converter = tf.lite.TFLiteConverter.from_saved_model('/Users/pratyushvivek/random_forest.joblib') # path to the SavedModel directory
tflite_model = converter.convert()'''
#converter = tf.lite.TFLiteConverter.from_keras_model(loaded_rf)
#tflite_model = converter.convert()
# Save the model.with open('Rf_model.tflite', 'wb') as f:
'''f.write(tflite_model)'''


def firebaseDownload(url):   
        try:
                cred = credentials.Certificate("./ndri-project-3f2d5-firebase-adminsdk-x4f9n-6f4eaaef3f.json")
                firebase_admin.initialize_app(cred,{"storageBucket":"ndri-project-3f2d5.appspot.com"})
        except Exception as e:
                print(e)
                pass

        bucket = storage.bucket()
        # blob = bucket.blob(url)
        blob = bucket.get_blob(url)
        # fileExist = blob.exists()
        # print(fileExist)
        arr = np.frombuffer(blob.download_as_string(),np.uint8)
        img = cv2.imdecode(arr,cv2.COLOR_BGR2BGR555)
        cv2.imwrite('image.jpg',img)
        # # cv2.imshow('image',img)
        # # cv2.waitKey(0)
        result = start()
        return result

# firebaseDownload("e82d771c-3a0f-4554-82fc-004fdf5faf601611877690209488383.jpg")