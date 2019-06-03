# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:26:09 2019

@author: Andri
"""

import numpy as np
import pandas as pd
import os
from LSTM_ln_class import LSTM_ln
from Train_test_datasets import Train_test
import warnings
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    scaler = StandardScaler()
    atributes = pd.read_csv('Atribute.csv',index_col=0)
    output = pd.read_csv('Output.csv',index_col=0) 
    output_ext = pd.read_csv('Output_ext.csv', index_col = 0)
    name1 = 'graph_1'
    
    stdev_train = np.load('stdev_train.npy')
    stdev_test = np.load('stdev_test.npy')
    stdev = np.concatenate((stdev_train,stdev_test))
    stdev = stdev[:-1]
    
    atributes = atributes.values
    new_atr = np.zeros([atributes.shape[0], stdev_train.shape[1]])
    new_atr[-stdev.shape[0]:] = stdev
    atributes[:,-new_atr.shape[1]:] = new_atr
    
    scaler.fit(atributes)
    atributes = scaler.transform(atributes)
   
    num_unroll = 320
    test_size1 = 0.2
    test_size2 = 0.35
    no_epoch = 120
    
    train1 = Train_test(num_unroll,test_size1,test_size2)
    _, test_X, _, _, _, _, _, _ = train1.train_test_fun(atributes, output.values)

    train_y_var = stdev_train
    test_y = stdev_test
      
    train_y_var = train_y_var[num_unroll:,:]
    
    scaler2 = StandardScaler()
    scaler2.fit(train_y_var)
    train_y_var = scaler2.transform(train_y_var)
    test_y = scaler2.transform(test_y)
    _, in_size = atributes.shape
    _, out_size = output.shape
    directory = os.getcwd()
    
    model1 = LSTM_ln(input_size=64, output_size=out_size, lstm_size=64, num_layers=3,
                             num_steps=num_unroll, keep_prob=0.8, batch_size=256, init_learning_rate=0.15,
                             learning_rate_decay=0.95, init_epoch=7, max_epoch = no_epoch, MODEL_DIR = directory, name = name1)
    
    model1.graph_name = 'graph_1_lr0.15_lr_decay0.950_lstm64_step320_input92_batch256_epoch120_kp0.800_layer3'
    prediction_test, _ = model1.prediction_by_trained_graph(no_epoch,test_X, test_y)
    
    prediction_test = scaler2.inverse_transform(prediction_test)
    
    np.save('stdev_test_predicted',prediction_test)
    
    
    
    
    
    
    
    