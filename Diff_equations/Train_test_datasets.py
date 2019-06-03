# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:54:06 2019

@author: Andri
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Train_test():
    
    def __init__(self,num_unroll, test_size_1 = 0.2, test_size_2 = 0.3):
        self.test_size_1 = test_size_1
        self.test_size_2 = test_size_2
        self.num_unroll = num_unroll
    
    @staticmethod
    def num_unroll_fun(data, data_out, num_unroll): 
        new_data = [data[i-num_unroll:i,:] for i in range(num_unroll,data.shape[0])]    
        new_data_out = data_out[num_unroll:]
        return new_data_out, new_data
    
    
    def train_test_fun(self, atributes, output):
        output, atributes = Train_test.num_unroll_fun(atributes.values, output.values, self.num_unroll)
        atributes = np.array(atributes)
        
        train_X, test_X, train_y, test_y = train_test_split(atributes, output, shuffle = False, test_size = self.test_size_1)
        
        train_X_mean, train_X_var, train_y_mean, train_y_var = train_test_split(train_X, train_y, shuffle = False, test_size = self.test_size_2)
        
        return train_X, test_X, train_y, test_y, train_X_mean, train_X_var, train_y_mean, train_y_var

if __name__ == '__main__':   
    atributes = pd.read_csv('Atribute.csv',index_col=0)
    output = pd.read_csv('Output.csv',index_col=0) 
    num_unroll = 200
    test_size1 = 0.2
    test_size2 = 0.4
    train1 = Train_test(num_unroll,test_size1,test_size2)
    train_X, test_X, train_y, test_t, train_X_mean, train_X_var, train_y_mean, train_y_var = train1.train_test_fun(atributes, output)