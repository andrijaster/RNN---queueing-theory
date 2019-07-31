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
    c_no = 7
    vreme2 = np.linspace(0,300,c_no)
       
    simulation_no = 10
    Xinit = np.zeros(broj_mesta)
    Xinit[0] = 1
    var_init = Xinit
    no = 1000
    c_var = np.average(C_var, axis=0)
    c_var_round = np.round(c_var)
    c_var_old = c_var_old[no,:]
    
    Lambda_real = y_real[no,:]
    Lambda_predicted = y_predicted[no,:]
    
    model1 = Optimal_control_discretize(broj_mesta, Lambda_predicted, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model1.evaluate_function_round(var_init, fun, c_var_round, c_var.shape[0])
    model2 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model2.evaluate_function_round(var_init, fun, c_var_round, c_var_round.shape[0])
    model5 = Optimal_control_discretize(broj_mesta, Lambda_real, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model5.evaluate_function(var_init, fun, c_var_old, c_var_old.shape[0])
    
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
        model7.evaluate_function_round(var_init, fun, c_var_round, c_var_round.shape[0])
        X_1[j,:,:] = model6.x
        X_2[j,:,:] = model7.x
        
        
    cmap = plt.get_cmap('jet_r')
    color = [cmap(float(i)/6) for i in range(3)]

    strs = ["$P_{%.d}$" % (float(x)) for x in range(3)]
    figure1 = plt.figure(figsize=(13, 4))
    ax1 = plt.subplot(111)
    for i in range(3):
        ax1.plot(model2.time_control, model2.x[:,i], c = color[i], lw =1.5)
        ax1.plot(model1.time_control, model1.x[:,i], c = color[i], lw =0.5)
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    ax1.set_ylabel('$P_i$')
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    ax1.set_xlabel('$t$ [min]')
    ax1.grid()
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,300)    
    plt.savefig('Prob_plot.png')
    
    
    figure2 = plt.figure(figsize=(13, 9))    
    ax1 = plt.subplot(211)
    ax1.step(vreme2, c_var_round, 'black', where = 'post')
    ax1.plot(vreme, c_var_old, 'red', lw = 0.5)
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
    
    
    figure1 = plt.figure(figsize=(13, 9))
    cmap = plt.get_cmap('jet_r')
    color = [cmap(float(i)/broj_mesta) for i in range(3)]
    X_1_avg = np.average(X_1,axis=0)
    X_2_avg = np.average(X_2,axis=0)

    for i in range(simulation_no):  
        ax1 = plt.subplot(211)
        for j in range(3):
            ax1.plot(model2.time_control, X_1[i,:,j], c=color[j],lw =0.5)
        ax1.set_ylim(0,1)
        ax1.set_xlim(0,model2.time_control[-1])
        ax1.set_ylabel('$P_i$')
        ax1.grid()
        ax1.legend(strs, ncol = 4, loc = 'upper right')
        
        ax2 = plt.subplot(212)
        for j in range(3):
            ax2.plot(model2.time_control, X_2[i,:,j], c=color[j], lw =0.5)
        ax2.set_ylim(0,1)
        ax2.set_xlim(0,model2.time_control[-1])
        ax2.set_ylabel('$P_i$')
        ax2.set_xlabel('$t$ [min]')
        ax2.legend(strs, ncol = 4, loc = 'upper right')
        
    for j in range(3):
        ax1.plot(model2.time_control, X_1_avg[:,j], c=color[j], lw=1.5)
        ax2.plot(model2.time_control, X_2_avg[:,j], c=color[j], lw=1.5)
        
    ax2.grid()
    ax1.grid()
        
        
    
    
        
    