# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:51:57 2019

@author: Andri
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from GRU_class import GRU
from Train_test_datasets import train_X_mean, train_X_var, train_y_mean, train_y_var, test_X, test_y
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

atributes = pd.read_csv('Atribute.csv',index_col=0)
output = pd.read_csv('Output.csv',index_col=0)



if __name__ == "__main__":
    
    steps_size = np.array([2**i for i in range(10)])
    steps_layers = np.arange(1,5)
    R2_score_1 = np.zeros([len(steps_layers), len(steps_size)])
    R2_score_2 = np.zeros([len(steps_layers), len(steps_size)])
    i=0
    j=0
    
    
    for layers in steps_layers:
        
        for size in steps_size:
        
            name1 = 'graph_1'
            name2 = 'graph_2'
           
            _, in_size = atributes.shape
            _, out_size = output.shape
            no_epoch = 400
            directory = os.getcwd()
    
            model1 = GRU(input_size=in_size, output_size=out_size, gru_size=size, num_layers=layers,
                             num_steps=1, keep_prob=0.8, batch_size=16, init_learning_rate=0.7,
                             learning_rate_decay=0.99, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name1)
            model2 = GRU(input_size=in_size, output_size=out_size, gru_size=size, num_layers=layers,
                             num_steps=1, keep_prob=1, batch_size=16, init_learning_rate=0.7,
                             learning_rate_decay=0.99, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name2)
            
            model1.build_gru_graph_with_config()
            model1.train_gru_graph(train_X_mean, train_y_mean, train_X_var, train_y_var)
            prediction,loss = model1.prediction_by_trained_graph(no_epoch,test_X, test_y)
            
            model2.build_gru_graph_with_config()
            model2.train_gru_graph(train_X_mean, train_y_mean, train_X_var, train_y_var)
            prediction2,loss2 = model2.prediction_by_trained_graph(no_epoch,test_X, test_y)
            
            R2_score_1[i,j] = r2_score(train_y_var,prediction)
            R2_score_2[i,j] = r2_score(train_y_var,prediction)
            
            j+=1
        i+=1