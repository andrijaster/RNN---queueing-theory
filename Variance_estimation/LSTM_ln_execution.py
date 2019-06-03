# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:51:57 2019

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
    train1 = Train_test(num_unroll,test_size1,test_size2)
    _, test_X, _, _, _, train_X_var, _, _ = train1.train_test_fun(atributes, output.values)

    train_y_var = stdev_train
    test_y = stdev_test
      
    train_X_var = train_X_var[num_unroll:,:,:]
    train_y_var = train_y_var[num_unroll:,:]
    
    scaler2 = StandardScaler()
    scaler2.fit(train_y_var)
    train_y_var = scaler2.transform(train_y_var)
    test_y = scaler2.transform(test_y)
    
    steps_size = np.array([2**i for i in range(4,8)])
    steps_layers = np.arange(1,4)
    
    R2_score_1 = np.zeros([len(steps_layers), len(steps_size)])
    R2_score_2 = np.zeros([len(steps_layers), len(steps_size)])
    MAE_1 = np.zeros([len(steps_layers), len(steps_size)])
    MAE_2 = np.zeros([len(steps_layers), len(steps_size)])
    MSE_1 = np.zeros([len(steps_layers), len(steps_size)])
    MSE_2 = np.zeros([len(steps_layers), len(steps_size)])
    i=0

    
    for layers in steps_layers:
        j=0        
        for size in steps_size:
        
            name1 = 'graph_1'
            name2 = 'graph_2'
           
            _, in_size = atributes.shape
            _, out_size = output.shape
            no_epoch = 120
            directory = os.getcwd()
    
            model1 = LSTM_ln(input_size=in_size, output_size=out_size, lstm_size=size, num_layers=layers,
                             num_steps=num_unroll, keep_prob=0.8, batch_size=256, init_learning_rate=0.15,
                             learning_rate_decay=0.96, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name1)
            model2 = LSTM_ln(input_size=in_size, output_size=out_size, lstm_size=size, num_layers=layers,
                             num_steps=num_unroll, keep_prob=1, batch_size=256, init_learning_rate=0.15,
                             learning_rate_decay=0.96, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name2)
            
            model1.build_lstm_graph_with_config()
            model1.train_lstm_graph(train_X_var, train_y_var, test_X, test_y)
            prediction,loss = model1.prediction_by_trained_graph(no_epoch,test_X, test_y)
            
            model2.build_lstm_graph_with_config()
            model2.train_lstm_graph(train_X_var, train_y_var, test_X, test_y)
            prediction2,loss2 = model2.prediction_by_trained_graph(no_epoch,test_X, test_y)
                      
            R2_score_1[i,j] = r2_score(test_y, prediction)            
            R2_score_2[i,j] = r2_score(test_y, prediction2)
            MAE_1[i,j] = mean_absolute_error(test_y, prediction)
            MAE_2[i,j] = mean_absolute_error(test_y, prediction2)
            MSE_1[i,j] = mean_squared_error(test_y, prediction)
            MSE_2[i,j] = mean_squared_error(test_y, prediction2)
            
            np.savetxt('R2_score_var_1.csv', R2_score_1)
            np.savetxt('R2_score_var_2.csv', R2_score_2)
            np.savetxt('MAE_var_1.csv', MAE_1)
            np.savetxt('MAE_var_2.csv', MAE_2)
            np.savetxt('MSE_var_1.csv', MSE_1)
            np.savetxt('MSE_var_2.csv', MSE_2)    
            
            j+=1
        i+=1
        