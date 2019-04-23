# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:06:07 2019

@author: Andri
"""

from Integrate import Maximum_principle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


""" Data """
def fun(i,price_minute):
    f = price_minute**i*i
    return f
    
    
output = pd.read_csv("Output_ext.csv", index_col = 0) 
mean_Lambda = output['lambda'].values[:200] # Predicted lambda
stand_dev = np.random.rand(mean_Lambda.shape[0]) # Predicted stddev
real_Lambda = output['lambda'].values[:200] + 0.01 # Real lambda

""" Initialize data """
simulation_no = 5
Lambda_matrix = np.zeros([simulation_no,mean_Lambda.shape[0]])

for i in range(mean_Lambda.shape[0]):
    Lambda_matrix[:,i] = np.random.normal(mean_Lambda[i], stand_dev[i], size = simulation_no)


vreme = np.linspace(0,199*5,200)
time_control = np.linspace(0,400,10000)
broj_mesta = 15
mu_val = 4.05103447
Cc = 30
price_minute = 3
c_var_pos = np.arange(1,12)

""" Initialize solutions """
c_var = np.zeros([simulation_no, time_control.shape[0]])
x = np.zeros([simulation_no, time_control.shape[0], broj_mesta])  
    
Xinit = np.zeros(broj_mesta)
t_x_init = np.zeros(broj_mesta)
Xinit[0] = 1
var_init = Xinit
time_init = t_x_init

for i in range(simulation_no):
    model1 = Maximum_principle(broj_mesta, Lambda_matrix[i,:], vreme, time_control, Cc, mu_val, price_minute, c_var_pos)
    model1.Optimal_control(var_init,time_init,fun)
    c_var[i,:] = model1.c_var_2
    x[i,:,:] = model1.x_2


model2 = Maximum_principle(broj_mesta, mean_Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_pos)
model2.Optimal_control(var_init,time_init,fun)


model3 = Maximum_principle(broj_mesta, real_Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_pos)
model3.Optimal_control(var_init,time_init,fun)


c_var_avg = np.mean(c_var,axis = 0)

c_var_avg_round = np.round(c_var_avg)
c_var_mean_round = np.round(model2.c_var_2)
c_var_real_round = np.round(model3.c_var_2)

prob_bayes, values = model1.Evaluate_values(var_init,c_var_avg,fun)
prob_bayes_real, values_real = model3.Evaluate_values(var_init,c_var_avg,fun)
prob_mean_real, values_mean_real = model3.Evaluate_values(var_init,model2.c_var_2,fun)

prob_bayes_real_round, values_bayes_real_round = model3.Evaluate_values(var_init,c_var_avg_round,fun)
prob_mean_real_round, values_mean_real_round = model3.Evaluate_values(var_init,c_var_mean_round,fun)
prob_real_round, values_real_round = model3.Evaluate_values(var_init,c_var_real_round,fun)

values_bayes = np.trapz(values,time_control)
values_mean = np.trapz(model2.values_2, time_control)
values_real_bayes = np.trapz(values_real, time_control)
values_real_mean = np.trapz(values_mean_real, time_control)
values_optimal = np.trapz(model3.values_2, time_control)

values_real_bayes_round = np.trapz(values_bayes_real_round, time_control)
values_real_mean_round = np.trapz(values_mean_real_round, time_control)
values_optimal_round = np.trapz(values_real_round, time_control)

print("Total_value_bayes = {}".format(values_bayes))
print("Total_value_mean = {}".format(values_mean))
print("Total_value_real_bayes = {}".format(values_real_bayes))
print("Total_value_real_mean = {}".format(values_real_mean))
print("Optimal_value_real_mean = {}".format(values_optimal))
print("Total_value_real_bayes_round = {}".format(values_real_bayes_round))
print("Total_value_real_mean_round = {}".format(values_real_mean_round))
print("Optimal_value_real_mean_round = {}".format(values_optimal_round))


strs = ["$P_{%.d}$" % (float(x)) for x in range(broj_mesta)]
figure1 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(211)
for i in range(simulation_no):
    ax1.plot(model1.time_control,c_var[i,:])
ax1.set_ylim(0,c_var_pos[-1]+1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$c(t)$')
ax1.grid()


ax1 = plt.subplot(212)
ax1.plot(model1.time_control,c_var_avg, 'black')
ax1.plot(model1.time_control,model2.c_var_2, 'red')
ax1.plot(model1.time_control,model3.c_var_2, 'blue', ls = '--')
ax1.set_ylim(0,c_var_pos[-1]+1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$c(t)$')
ax1.grid()
ax1.set_xlabel('$t$ [min]')
ax1.legend(["Bayes", "Mean", "Optimal"], loc = 'upper right')

figure2 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control,prob_bayes)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax1 = plt.subplot(312)
ax1.plot(model1.time_control,model2.x_2)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax1 = plt.subplot(313)
for i in range(simulation_no):
    ax1.plot(model1.time_control,x[i,:,:])
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.set_xlabel('$t$ [min]')
ax1.legend(strs, ncol = 3, loc = 'upper right')

figure3 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control,prob_bayes_real)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax1 = plt.subplot(312)
ax1.plot(model1.time_control,prob_mean_real)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax1 = plt.subplot(313)
ax1.plot(model1.time_control,model3.x_2)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.set_xlabel('$t$ [min]')
ax1.legend(strs, ncol = 3, loc = 'upper right')

figure4 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control,values_real)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$Value$ [EUR/minute]' )
ax1.grid()

ax1 = plt.subplot(312)
ax1.plot(model1.time_control,values_mean_real)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$Value$ [EUR/minute]')
ax1.grid()

ax1 = plt.subplot(313)
ax1.plot(model1.time_control,model3.values_2)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$Value$ [EUR/minute]')
ax1.grid()
ax1.set_xlabel('$t$ [min]')

figure5 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control, c_var_avg_round)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$c(t)$')
ax1.grid()

ax1 = plt.subplot(312)
ax1.plot(model1.time_control,c_var_mean_round)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$c(t)$')
ax1.grid()

ax1 = plt.subplot(313)
ax1.plot(model1.time_control,c_var_real_round)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$c(t)$')
ax1.grid()
ax1.set_xlabel('$t$ [min]')

figure6 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control,values_bayes_real_round)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$Value$ [EUR/minute]' )
ax1.grid()

ax1 = plt.subplot(312)
ax1.plot(model1.time_control,values_mean_real_round)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$Value$ [EUR/minute]')
ax1.grid()

ax1 = plt.subplot(313)
ax1.plot(model1.time_control,values_real_round)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$Value$ [EUR/minute]')
ax1.grid()
ax1.set_xlabel('$t$ [min]')

figure7 = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control, prob_bayes_real_round)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax1 = plt.subplot(312)
ax1.plot(model1.time_control,prob_mean_real_round)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax1 = plt.subplot(313)
ax1.plot(model1.time_control,prob_real_round)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.grid()
ax1.set_xlabel('$t$ [min]')
ax1.legend(strs, ncol = 3, loc = 'upper right')

