#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:32:16 2018

@author: alper
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np

## fix random seed for reproducibility
np.random.seed(7)

#load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv",delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
 
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, update_freq='epoch')
#Fit the model
history = model.fit(X,Y,epochs=150,batch_size=10,callbacks=[tbCallBack])


plt.plot(history.history['loss'])    
#Evaluate the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
