# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:37:40 2019

@author: Andri
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:16:22 2019

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
    
    
    y_real = np.load('y_test_predict_real.npy')
    y_predicted = np.load('y_test_predict.npy')
    stdev_predicted = np.load('stdev_test_predicted.npy')
    c_var = np.load('c_var_300.npy')
    c_var_round = np.load('c_var_round_300.npy')
    c_var_old = pd.read_csv('Track_no.csv', header = 0, index_col = 0)
    c_var_old = c_var_old.values[-y_real.shape[0]:]
        
    vreme = np.linspace(0, 300, y_real.shape[1])
    time_control = np.linspace(0,300,1001)
    broj_mesta = 12
    c_var_min = 1
    c_var_max = 13
    mu_val = 4.05103447
    Cc = 0.162
    price_minute = 0.153
    c_no = 300
    vreme2 = np.linspace(0,300,c_no)
       
    Xinit = np.zeros(broj_mesta)
    Xinit[0] = 1
    var_init = Xinit
    no = 1000
    c_var_old = c_var_old[no,:]
    
    
    Lambda_real = y_real[no,:]
    Lambda_predicted = y_predicted[no,:]
    
    model1 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model1.evaluate_function(var_init, fun, c_var, c_var.shape[0])
    model2 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model2.evaluate_function_round(var_init, fun, c_var_round, c_var_round.shape[0])
    

    model3 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model3.evaluate_function(var_init, fun, c_var, c_var.shape[0])
    model4 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model4.evaluate_function_round(var_init, fun, c_var_round, c_var_round.shape[0])
    
    model5 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model5.evaluate_function(var_init, fun, c_var_old, c_var_old.shape[0])
    model6 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model6.evaluate_function(var_init, fun, c_var_old, c_var_old.shape[0])
    
    
    cmap = plt.get_cmap('jet_r')
    color = [cmap(float(i)/6) for i in range(3)]

    strs = ["$P_{%.d}$" % (float(x)) for x in range(3)]
    figure1 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    for i in range(3):
        ax1.plot(model1.time_control, model1.x[:,i], c = color[i], lw =1.5)
        ax2.plot(model1.time_control, model2.x[:,i], c = color[i], lw =1.5)
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    ax2.legend(strs, ncol = 4, loc = 'upper right') 
    for i in range(3):
        ax1.plot(model1.time_control, model3.x[:,i], c = color[i], lw = 0.5)
        ax2.plot(model1.time_control, model4.x[:,i], c = color[i], lw = 0.5)        
    ax1.set_ylabel('$P_i$')
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    ax2.set_ylabel('$P_i$')
    ax2.legend(strs, ncol = 4, loc = 'upper right')
    ax2.set_xlabel('$t$ [min]')
    ax1.grid()
    ax2.grid()
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    ax1.set_xlim(0,300)
    ax2.set_xlim(0,300)
    
    plt.savefig('Prob_plot.png')
    

    figure2 = plt.figure(figsize=(13, 9))    
    ax1 = plt.subplot(211)
    ax1.step(vreme2, c_var_round, 'black', where = 'post')
    ax1.plot(vreme,c_var_old, 'red', lw = 0.5)
    ax1.set_ylim(0,13)
    ax1.set_xlim(0,model1.time_control[-1])
    ax1.set_ylabel('$c(t)$')
    ax1.grid()
    ax1.legend(['оптимално','тренутно'], loc = 'upper right')
    
    ax2 = plt.subplot(212)
    ax2.plot(model2.time_control, model2.values_eva, lw = 1.5)
    ax2.plot(model5.time_control, model5.values_eva, lw = 0.5)
    ax2.set_ylim(0,np.max(model1.values_eva)+2)
    ax2.set_xlim(0,model1.time_control[-1])
    ax2.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax2.set_xlabel('$t$ [min]')
    ax2.grid()
    ax2.legend(['оптимално','тренутно'], loc = 'upper right')
    
    plt.savefig('c(t)-E(UT).png')
    
    figure3 = plt.figure(figsize=(13, 9))    
    ax1 = plt.subplot(211)
    ax1.plot(vreme2, c_var, 'black', lw = 1.5)
    ax1.plot(vreme,c_var_old, 'red', lw = 0.5)
    ax1.set_ylim(0,13)
    ax1.set_xlim(0,model1.time_control[-1])
    ax1.set_ylabel('$c(t)$')
    ax1.grid()
    ax1.legend(['оптимално','тренутно'], loc = 'upper right')
    
    ax2 = plt.subplot(212)
    ax2.plot(model1.time_control, model1.values_eva, lw = 1.5)
    ax2.plot(model5.time_control, model5.values_eva, lw = 0.5)
    ax2.set_ylim(0,np.max(model1.values_eva)+2)
    ax2.set_xlim(0,model1.time_control[-1])
    ax2.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax2.set_xlabel('$t$ [min]')
    ax2.grid()
    ax2.legend(['оптимално','тренутно'], loc = 'upper right')
    
    plt.savefig('c(t)-E(UT).png')