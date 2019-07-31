# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:40:25 2019

@author: Andri
"""

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
    sample = np.random.choice(y_real.shape[0], 250)
    simulation_no =10
    
    total_val_old = np.zeros(sample.shape[0])
    total_val_round = np.zeros(sample.shape[0])
    
    j=0
    for no in sample:
        Lambda_predict = y_predicted[no,:]
        model1 = Optimal_control_discretize(broj_mesta, Lambda_predict, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')        
        model1.Optimal_control_round(var_init, fun, c_no)
        
        Lambda_real= y_real[no,:]
        model2 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')        
        model2.evaluate_function(var_init, fun, model1.c_var_round, model1.c_var_round.shape[0])
        model3 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')        
        model3.evaluate_function(var_init, fun, c_var_old[no,:], model1.c_var_round.shape[0])
        
        total_val_old[j] = model3.tot_value_eva
        total_val_round[j] = model2.tot_value_eva     
        j+=1
    
    avg_value_old = np.average(total_val_old)   
    avg_value = np.average(total_val_round) 
    
    np.save('tot_value_round', total_val_round)
    np.save('tot_value_old', total_val_old)
    np.save('avg_value', avg_value)
    np.save('avg_value_old', avg_value_old)