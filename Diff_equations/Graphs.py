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
    c_var = np.load('c_var.npy')
    c_var_round = np.load('c_var_round.npy')
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
    model2.evaluate_function(var_init, fun, c_var_round, c_var_round.shape[0])
    

    model3 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model3.evaluate_function(var_init, fun, c_var, c_var.shape[0])
    model4 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model4.evaluate_function(var_init, fun, c_var_round, c_var_round.shape[0])
    
    model5 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model5.evaluate_function(var_init, fun, c_var_old, c_var_old.shape[0])
    model6 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model6.evaluate_function(var_init, fun, c_var_old, c_var_old.shape[0])
    
    """ Model 1 i 2 """
    strs = ["$P_{%.d}$" % (float(x)) for x in range(model1.broj_mesta)]
    figure1 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model1.time_control, model1.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model1.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.plot(vreme2,c_var, 'black')
    ax2.set_ylim(0,13)
    ax2.set_xlim(0,model1.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()
    ax2.legend('upravljanje',  loc = 'upper right')
    
    ax3 = plt.subplot(313)
    ax3.plot(vreme2, model1.values_eva, lw = 5)
    ax3.set_ylim(0,np.max(model1.values_eva)+2)
    ax3.set_xlim(0,model1.time_control[-1])
    ax3.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()
    ax3.legend('vrednosti',  loc = 'upper right')
    
    plt.savefig('model1.png')
    
    strs = ["$P_{%.d}$" % (float(x)) for x in range(model1.broj_mesta)]
    figure1 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model2.time_control, model2.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model2.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.step(vreme2,c_var_round, 'black', where='post')
    ax2.set_ylim(0,13)
    ax2.set_xlim(0,model2.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()
    ax2.legend('upravljanje',  loc = 'upper right')
    
    ax3 = plt.subplot(313)
    ax3.plot(vreme2,model2.values_eva, lw = 5)
    ax3.set_ylim(0,np.max(model2.values_eva)+2)
    ax3.set_xlim(0,model2.time_control[-1])
    ax3.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()
    ax3.legend('vrednosti',  loc = 'upper right')
    
    plt.savefig('model2.png')
    
    
    """ Model 3 i 4 """
    strs = ["$P_{%.d}$" % (float(x)) for x in range(model3.broj_mesta)]
    figure3 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model3.time_control, model3.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model3.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.plot(vreme2,c_var, 'black')
    ax2.set_ylim(0,13)
    ax2.set_xlim(0,model3.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()
    ax2.legend('upravljanje',  loc = 'upper right')
    
    ax3 = plt.subplot(313)
    ax3.plot(vreme2, model3.values_eva, lw = 5)
    ax3.set_ylim(0,np.max(model3.values_eva)+2)
    ax3.set_xlim(0,model3.time_control[-1])
    ax3.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()
    ax3.legend('vrednosti',  loc = 'upper right')
    
    plt.savefig('model3.png')
    
    strs = ["$P_{%.d}$" % (float(x)) for x in range(model4.broj_mesta)]
    figure4 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model4.time_control, model4.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model4.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.step(vreme2,c_var_round, 'black', where='post')
    ax2.set_ylim(0,13)
    ax2.set_xlim(0,model4.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()
    ax2.legend('upravljanje',  loc = 'upper right')
    
    ax3 = plt.subplot(313)
    ax3.plot(vreme2,model4.values_eva, lw = 5)
    ax3.set_ylim(0,np.max(model4.values_eva)+2)
    ax3.set_xlim(0,model4.time_control[-1])
    ax3.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()
    ax3.legend('vrednosti',  loc = 'upper right')
    
    plt.savefig('model4.png')
    
    """ Model 5 i 6 """
    strs = ["$P_{%.d}$" % (float(x)) for x in range(model5.broj_mesta)]
    figure5 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model5.time_control, model5.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model5.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.plot(vreme,c_var_old, 'black')
    ax2.set_ylim(0,13)
    ax2.set_xlim(0,model5.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()
    ax2.legend('upravljanje',  loc = 'upper right')
    
    ax3 = plt.subplot(313)
    ax3.plot(vreme, model5.values_eva, lw = 5)
    ax3.set_ylim(0,np.max(model5.values_eva)+2)
    ax3.set_xlim(0,model5.time_control[-1])
    ax3.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()
    ax3.legend('vrednosti',  loc = 'upper right')
    
    plt.savefig('model5.png')
    
    strs = ["$P_{%.d}$" % (float(x)) for x in range(model6.broj_mesta)]
    figure6 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model6.time_control, model6.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model6.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.plot(vreme,c_var_old, 'black')
    ax2.set_ylim(0,13)
    ax2.set_xlim(0,model6.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()
    ax2.legend('upravljanje',  loc = 'upper right')
    
    ax3 = plt.subplot(313)
    ax3.plot(vreme, model6.values_eva, lw = 5)
    ax3.set_ylim(0,np.max(model6.values_eva)+2)
    ax3.set_xlim(0,model6.time_control[-1])
    ax3.set_ylabel('$E[UT](t)$ [EUR/min]')
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()
    ax3.legend('vrednosti',  loc = 'upper right')
    
    plt.savefig('model6.png')
    
    
    