from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
from keras.models import Model
import os
import time
import torch
import numpy as np
import pandas as pd
import sys
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

def get_keras_model(lookback,n_features):
    model = Sequential()  
    model.add(LSTM(128,input_shape = (lookback, n_features), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(128,input_shape = (lookback, n_features)))
    model.add(Dense(1))
    print(model.summary())
    model.add(Activation('linear'))
    model.compile(loss = 'mse', optimizer = 'adam')
    return model

def train(model,lookback,n_features, x_train, y_train,x_val, y_val):
  
    earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='min',verbose=1,min_delta=0.00001)
    history = model.fit(x_train,y_train, epochs = 250, batch_size=48
          ,validation_data=(x_val,y_val),verbose = 1, 
          shuffle = False, callbacks=[earlystop])
    # history = model.fit(x_train,y_train, epochs = 100, batch_size=48
    #       ,validation_data=(x_val,y_val),verbose = 1, 
    #       shuffle = False)
    loss = history.history
    plt.plot(loss['loss'],label='loss')
    plt.plot(loss['val_loss'],label='vaild loss')
    plt.legend()
    plt.show()


def get_data(df,lookback,n_features):
    sc = MinMaxScaler()
    data = sc.fit_transform(df) 
    train_ind = int(0.6*len(df))
    val_ind = train_ind + int(0.2*len(df))
    train = data[:train_ind]
    val = data[train_ind:val_ind]
    test = data[val_ind:]
    xtrain,ytrain,xval,yval,xtest,ytest = train[:,:7],train[:,0],val[:,:7],val[:,0],test[:,:7],test[:,0]
    train_len = len(xtrain) - lookback
    test_len = len(xtest) - lookback
    val_len = len(xval) - lookback
    
    x_train = np.zeros((train_len, lookback, n_features))
    y_train = np.zeros((train_len))
    for i in range(train_len):
        ytemp = i+lookback
        x_train[i] = xtrain[i:ytemp]
        y_train[i] = ytrain[ytemp]
            
    x_val = np.zeros((val_len, lookback, n_features))
    y_val = np.zeros((val_len))
    for i in range(val_len):
        ytemp = i+lookback
        x_val[i] = xval[i:ytemp]
        y_val[i] = yval[ytemp]
        
    x_test = np.zeros((test_len, lookback, n_features))
    y_test = np.zeros((test_len))
    for i in range(test_len):
        ytemp = i+lookback
        x_test[i] = xtest[i:ytemp]
        y_test[i] = ytest[ytemp]
    
    return x_train, x_val, y_train, y_val,x_test,y_test
    
        

        