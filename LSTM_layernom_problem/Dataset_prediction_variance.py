# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:07:42 2019

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
    
    scaler.fit(atributes)
    atributes = scaler.transform(atributes)
    name1 = 'graph_1'
    
    num_unroll = 320
    test_size1 = 0.2
    test_size2 = 0.35
    train1 = Train_test(num_unroll,test_size1,test_size2)
    train_X, test_X, train_y, test_y, train_X_mean, train_X_var, train_y_mean, train_y_var = train1.train_test_fun(atributes, output.values)
   
    output_ext = output_ext[num_unroll:]
    help_x1, test_x_ext, help_y1, test_y_ext = train_test_split(atributes[num_unroll:], output_ext, shuffle = False, test_size = test_size1)
    _, _, _, train_y_var_ext = train_test_split(help_x1, help_y1, shuffle = False, test_size = test_size2)
    
    test_y_check = test_y_ext.values.cumsum(axis = 1)
    test_y_check = test_y_check[:,1:]
    
    train_y_check = train_y_var_ext.values.cumsum(axis = 1)
    train_y_check = train_y_check[:,1:]
    
    _, in_size = atributes.shape
    _, out_size = output.shape
    no_epoch = 120
    directory = os.getcwd()
    
    model1 = LSTM_ln(input_size=in_size, output_size=out_size, lstm_size=32, num_layers=3,
                             num_steps=num_unroll, keep_prob=0.8, batch_size=256, init_learning_rate=0.15,
                             learning_rate_decay=0.95, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name1)

#    model1.build_lstm_graph_with_config(
    model1.graph_name = 'graph_1_lr0.15_lr_decay0.950_lstm32_step320_input92_batch256_epoch120_kp0.800_layer3'
    prediction_test, _ = model1.prediction_by_trained_graph(no_epoch,test_X, test_y)
    prediction_train, _ = model1.prediction_by_trained_graph(no_epoch,train_X_var, train_y_var)
    
    prediction_test = np.insert(prediction_test,0,test_y_ext.values[:,0], axis=1) 
    prediction_test = prediction_test.cumsum(axis = 1)
    prediction_test = prediction_test[:,1:]
            
    prediction_train = np.insert(prediction_train,0,train_y_var_ext.values[:,0], axis=1) 
    prediction_train = prediction_train.cumsum(axis = 1)
    prediction_train = prediction_train[:,1:]    
    
    stdev_train = np.abs(prediction_train - train_y_check)
    stdev_test = np.abs(prediction_test - test_y_check)

    np.save('stdev_train', stdev_train)
    np.save('stdev_test', stdev_test)
    np.save('y_test_predict', prediction_test)
    np.save('y_test_predict_real', test_y_check)
