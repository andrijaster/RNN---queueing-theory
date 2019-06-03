# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:12:37 2019

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
    stdev = np.load('stdev_test_predicted.npy')
    C_var = np.load('C_var_round_BAYES.npy')

    vreme = np.linspace(0, 300, y_real.shape[1])
    time_control = np.linspace(0,300,1001)
    broj_mesta = 12
    c_var_min = 1
    c_var_max = 13
    mu_val = 4.05103447
    Cc = 0.162
    price_minute = 0.153
    c_no = 7
    vreme2 = np.linspace(0,300,c_no)
       
    simulation_no = 10
    Xinit = np.zeros(broj_mesta)
    Xinit[0] = 1
    var_init = Xinit
    no = 1000
    c_var = np.average(C_var, axis=0)
    c_var_round = np.round(c_var)
    
    Lambda_real = y_real[no,:]
    
    model1 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model1.evaluate_function(var_init, fun, c_var, c_var.shape[0])
    model2 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model2.evaluate_function(var_init, fun, c_var_round, c_var_round.shape[0])
    
    mean_Lambda = y_predicted[no,:]
    stand_dev = stdev[no,:]
    Lambda_matrix = np.zeros([simulation_no,mean_Lambda.shape[0]])
    X_1 = np.zeros([simulation_no, time_control.shape[0], broj_mesta])  
    X_2 = np.zeros([simulation_no, time_control.shape[0], broj_mesta])
    
    for i in range(mean_Lambda.shape[0]):
        Lambda_matrix[:,i] = np.random.normal(mean_Lambda[i], stand_dev[i], size = simulation_no)

    for j in range(simulation_no):
        Lambda = Lambda_matrix[j,:]
        model6 = Optimal_control_discretize(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
        model6.evaluate_function(var_init, fun, c_var, c_var.shape[0])
        model7 = Optimal_control_discretize(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
        model7.evaluate_function(var_init, fun, c_var_round, c_var_round.shape[0])
        X_1[j,:,:] = model6.x
        X_2[j,:,:] = model7.x
    
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
    
    plt.savefig('model1-B.png')
    
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
    
    plt.savefig('model2-B.png')
    
    
    figure1 = plt.figure(figsize=(13, 9))
    cmap = plt.get_cmap('jet_r')
    color = [cmap(float(i)/broj_mesta) for i in range(broj_mesta)]
    X_1_avg = np.average(X_1,axis=0)
    X_2_avg = np.average(X_2,axis=0)

    for i in range(simulation_no):  
        ax1 = plt.subplot(211)
        for j in range(broj_mesta):
            ax1.plot(model2.time_control, X_1[i,:,j], c=color[j],lw =0.5)
        ax1.set_ylim(0,1)
        ax1.set_xlim(0,model2.time_control[-1])
        ax1.set_ylabel('$P_i$')
        ax1.grid()
        ax1.legend(strs, ncol = 4, loc = 'upper right')
        
        ax2 = plt.subplot(212)
        for j in range(broj_mesta):
            ax2.plot(model2.time_control, X_2[i,:,j], c=color[j], lw =0.5)
        ax2.set_ylim(0,1)
        ax2.set_xlim(0,model2.time_control[-1])
        ax2.set_ylabel('$P_i$')
        ax2.set_xlabel('$t$ [min]')
        ax2.legend(strs, ncol = 4, loc = 'upper right')
        ax2.grid()
        
    for j in range(broj_mesta):
        ax1.plot(model2.time_control, X_1_avg[:,j], c=color[j], lw=1.5)
        ax2.plot(model2.time_control, X_2_avg[:,j], c=color[j], lw=1.5)
        
        
    
    
        
    