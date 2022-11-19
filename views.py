# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 01:45:40 2022

@author: hp
"""

from django.shortcuts import render
from django.http import JsonResponse
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


df = pd.read_csv('titanic.csv')
#drop the name columns
df = df.drop(columns='Name')
#encode the sex column
df.loc[df['Sex']=='male', 'Sex'] = 1
df.loc[df['Sex']=='female', 'Sex'] = 0
#split the data in to independent x and y variables
X = df.drop('Survived', axis =1)
y=df['Survived'].values.astype(np.float32)
X = X.values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=1)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(6, activation=tf.nn.relu),
	keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']) 


history = model.fit(X_train, y_train, epochs=119, batch_size=1,validation_data=(X_val, y_val))


model = tf.keras.models.load_model('titanic.h5')
model.load_weights('model.h5')
model.summary()
model.get_weights()


# our home page view
def home(request):    
    print(request.POST)
    pclass = int(request.POST.get('pclass', False))
    sex = int(request.POST.get('sex', False))
    age = int(request.POST.get('age', False))
    n_siblings_spouses = int(request.POST.get('n_siblings_spouses', False))
    n_parents_children = int(request.POST.get('n_parents_children', False))
    fare = int(request.POST.get('fare', False))
    
    data = [[pclass,sex, age, n_siblings_spouses,n_parents_children,fare]]
    data = tf.constant(data)
    #data=np.float32(data)
    prediction = model.predict(data, steps=1)
    pred = [round(x[0]) for x in prediction]
    if pred == [0]:
        result = 'You did not survive!'
    elif pred == [1]:
        result = 'You survived!'
    else: 
        result = 'Error!'''
        
    return render(request, 'index.html', {'result': result})
    #return render(request, 'index.html')

#def prediction(request):
    
    
    #return render(request, 'index.html', {'result':data})
    
    
    
    
    
    
