# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:54:06 2019

@author: Andri
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

atributes = pd.read_csv('Atribute.csv',index_col=0)
output = pd.read_csv('Output.csv',index_col=0)
train_X, test_X, train_y, test_y = train_test_split(atributes, output, shuffle = False, test_size = 0.2)

train_X_mean, train_X_var, train_y_mean, train_y_var = train_test_split(train_X, train_y, shuffle = False, test_size = 0.3)

train_X_mean = np.expand_dims(train_X_mean, axis = 1)
train_X_var = np.expand_dims(train_X_var, axis = 1)
test_X = np.expand_dims(test_X, axis = 1)