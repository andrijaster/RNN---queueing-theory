# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:28:32 2019

@author: Andri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.integrate import odeint
from Optimal_control_discretize import Optimal_control_discretize
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    def fun(i,price_minute):
        f = price_minute*i
        return f
    
    
    stdev_predicted = np.load('stdev_test_predicted.npy')
    stdev_real = np.load('stdev_test.npy')
    y_real = np.load('y_test_predict_real.npy')
    y_predicted = np.load('y_test_predict.npy')
    c_var_old = pd.read_csv('Track_no.csv', header = 0, index_col = 0)
    c_var_old = c_var_old.values[-stdev_real.shape[0]:]

    vreme = np.linspace(0,300,c_var_old.shape[1])
    time_control = np.linspace(0,300,1001)
    broj_mesta = 12
    mu_val = 4.05103447
    Cc = 0.162
    price_minute = 0.153
    c_var_min = 1
    c_var_max = 13
    c_no = 7
    vreme2 = np.linspace(0,300,c_no)
       
    Xinit = np.zeros(broj_mesta)
    Xinit[0] = 1
    var_init = Xinit
    no = 1000
    simulation_no =10
    
#    Lambda_predict = y_predicted[no,:]
#    model1 = Optimal_control_discretize(broj_mesta, Lambda_predict, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
#    model1.Optimal_control(var_init, fun, c_no)
#    model1.Optimal_control_round(var_init, fun, c_no)
#    np.save('c_var', model1.c_var)
#    np.save('c_var_round', model1.c_var_round)
    
    
    """ Initialize data """
    mean_Lambda = y_predicted[no,:]
    stand_dev = stdev_predicted[no,:]
    Lambda_matrix = np.zeros([simulation_no,mean_Lambda.shape[0]])
    C_var = np.zeros([simulation_no, c_no]) 
    x = np.zeros([simulation_no, time_control.shape[0], broj_mesta])  

    
    for i in range(mean_Lambda.shape[0]):
        Lambda_matrix[:,i] = np.random.normal(mean_Lambda[i], stand_dev[i], size = simulation_no)

    for j in range(simulation_no):
        Lambda = Lambda_matrix[j,:]
        model2 = Optimal_control_discretize(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
        model2.Optimal_control_round(var_init, fun, c_no)
        C_var[j,:] = model2.c_var_round
        model2.evaluate_function(var_init, fun, model2.c_var_round, model2.c_var_round.shape[0])
        x[j,:,:] = model2.x
        
    np.save('C_var_round_BAYES', C_var)
    np.save('X_BAYES',x)
    

    

    
    

    
    
    

