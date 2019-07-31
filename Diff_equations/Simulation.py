# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:00:21 2019

@author: Andri
"""

import numpy as np
import pandas as pd
from tick.hawkes import SimuInhomogeneousPoisson
from tick.base import TimeFunction
import matplotlib.pyplot as plt
from Optimal_control_discretize import Optimal_control_discretize


import scipy.stats as sp
import warnings
warnings.filterwarnings('ignore')

""" Interpolate between lambda """
def Lambda_value(t,vreme,Lambda):
    lambda_val = np.interp(t,vreme,Lambda)
    return lambda_val

""" Interpolate between c """
def C_value(t,vreme,c_var):
    c_val = np.round(np.interp(t,vreme,c_var))
    return c_val

def fun(i,price_minute):
    f = price_minute*i
    return f

class Simulation():

    """ Interpolate between lambdas """    
    @staticmethod
    def Lambda_interp(t,vreme,Lambda):
        lambda_val = np.interp(t,vreme,Lambda)
        return lambda_val
    
    """ Interpolate between C_opt """
    @staticmethod
    def C_interp(t, vreme_c, c_opt):
        c_new = np.round(np.interp(t, vreme_c, c_opt))
        return c_new    
    
    """ Generate point process """
    @staticmethod
    def generate_times_opt(max_t, delta, vreme_c, c_var, vreme, Lambda):
        time = np.arange(delta,max_t, delta)
        c_value_t = Simulation.C_interp(time,vreme_c,c_var)
        lamb_val_t = Simulation.Lambda_interp(time,vreme,Lambda)/c_value_t
        tf = TimeFunction((time, lamb_val_t), dt=delta)
        Psim = SimuInhomogeneousPoisson([tf], end_time = max_t, verbose = False)
        Psim.simulate()
        simulacija = Psim.timestamps 
        return simulacija[0]
        
    def __init__(self, no_sim, tsim, step_size, loc_exp, 
                 scale_exp, lambda_sim,vreme, c_opt, vreme_c, stanje):
        
        self.step_size = step_size
        self.no_sim = no_sim
        self.tsim = tsim
        self.Lambda= lambda_sim
        self.loc = loc_exp
        self.scale = scale_exp
        self.c_opt = c_opt
        self.vreme = vreme
        self.vreme_c = vreme_c
        self.state = stanje
        self.noinst = int(self.tsim // self.step_size)
        
    def sim(self):
        
        Nws = np.zeros([self.no_sim,self.noinst])+1
        Nw = np.zeros([self.no_sim,self.noinst])
        Probability = np.zeros([self.state, self.noinst])
        
        for i in range(self.no_sim):
            
            t_arrival = Simulation.generate_times_opt(self.tsim, self.step_size, self.vreme_c, 
                                           self.c_opt, self.vreme, self.Lambda) 
            t_arrival = np.round(t_arrival,2)
            dist = sp.expon(loc = self.loc, scale = self.scale)
            data1 = np.round(dist.rvs(t_arrival.shape),2)
            instance = (t_arrival/self.step_size).astype(int)
            instance_pom = (data1/self.step_size).astype(int)
            k = 0
            instance2 = 0
            for j in instance:
                if instance2<self.noinst:
                    if Nws[i,instance2] == 0:
                        instance2 = prethodna + instance_pom[k]
                        Nws[i,j:] += 1
                        Nws[i,instance2:] -= 1
                    else:
                        instance2 = instance2 + instance_pom[k]
                        Nws[i,j:] += 1
                        Nws[i,instance2:] -= 1
                        Nw[i,j:] += 1
                        Nw[i,instance2:] -= 1
                    prethodna = j
                    k+=1
        for ind in range(self.state):
            Nws_copy = Nws.copy()
            if ind<self.state-1 and ind>0:
                if Nws_copy[Nws_copy==ind].size > 0:
                    Nws_copy[Nws_copy!=ind] = 0
                    Nws_copy[Nws_copy==ind] = 1
            elif ind == 0:
                if Nws_copy[Nws_copy==ind].size > 0:
                    Nws_copy[Nws_copy!=ind] = 3
                    Nws_copy[Nws_copy==ind] = 1
                    Nws_copy[Nws_copy!=1] = 0
            else:
                if Nws_copy[Nws_copy==ind].size > 0:
                    Nws_copy = Nws.copy()
                    Nws_copy[Nws_copy<ind] = 0   
                    Nws_copy[Nws_copy>=ind] = 1
            pomocna = Nws_copy.copy()
            Probability[ind,:] = np.sum(pomocna,axis=0)/self.no_sim
        Nws_mean = np.mean(Nws,axis = 0)
        Nw = np.mean(Nw, axis = 0)
                
        return Nws_mean, Nws, Nw, Probability
                
                    
                          
                    
                        
                
if __name__ == '__main__':
    """ Data """
    y_real = np.load('y_test_predict_real.npy')
    c_var_round = np.load('c_var_round.npy')
    no = 1000
    Lambda = y_real[no,:]
    
    loc_exp = 0
    scale_exp = 0.24685052850561828
    vreme = np.linspace(0, 300, Lambda.shape[0])
    vreme_c = np.linspace(0,300,c_var_round.shape[0])
    state = 12
    
    model1 = Simulation(200,300,0.01,loc_exp, scale_exp, Lambda, vreme, c_var_round, vreme_c, state)
    Nws200, Nws200f, Nw200, Probability200 = model1.sim()
    model2 = Simulation(1000,300,0.01,loc_exp, scale_exp, Lambda, vreme, c_var_round, vreme_c, state)
    Nws1000, Nws1000f, Nw1000, Probability1000 = model2.sim()
    model3 = Simulation(3000,300,0.01,loc_exp, scale_exp, Lambda, vreme, c_var_round, vreme_c, state)
    Nws3000, Nws3000f, Nw3000, Probability3000 = model3.sim()
    model4 = Simulation(5000,300,0.01,loc_exp, scale_exp, Lambda, vreme, c_var_round, vreme_c, state)
    Nws5000, Nws5000f, Nw5000, Probability5000 = model4.sim()
    model5 = Simulation(10000,300,0.01,loc_exp, scale_exp, Lambda, vreme, c_var_round, vreme_c, state)
    Nws10000, Nws10000f, Nw10000, Probability10000 = model4.sim()
    t = np.linspace(0,300,Nws200.shape[0])
    

    
    vreme = np.linspace(0, 300, y_real.shape[1])
    time_control = np.linspace(0,300,Probability200.shape[1])
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
    
    
    model4 = Optimal_control_discretize(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max, method = 'differential_evolution')
    model4.evaluate_function_round(var_init, fun, c_var_round, c_var_round.shape[0])
    
    
    strs = ["$P_{%.d}$" % (float(x)) for x in range(3)]
        
    cmap = plt.get_cmap('jet_r')
    color = [cmap(float(i)/6) for i in range(3)]
    color = ['red','yellow','pink']
    
    strs = ["$P_{%.d}$" % (float(x)) for x in range(3)]
    figure1 = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    for i in range(3):
        ax1.plot(t, Probability200.T[:,i], c = color[i], lw =1.5)
        ax1.plot(model4.time_control, model4.x[:,i], c = 'black', lw =0.6)
    ax1.legend(strs, ncol = 4, loc = 'upper right')     
    ax1.set_ylabel('$P_i$')
    ax1.legend(strs, ncol = 4, loc = 'upper right')
    ax1.grid()
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,300)
    ax1.legend(strs, ncol = 3, loc = 'upper right')
    
    strs = ["$P_{%.d}$" % (float(x)) for x in range(3)]
    ax2 = plt.subplot(312)
    for i in range(3):
        ax2.plot(t, Probability1000.T[:,i], c = color[i], lw =1.5)
        ax2.plot(model4.time_control, model4.x[:,i], c = 'black', lw =0.6)
    ax2.legend(strs, ncol = 4, loc = 'upper right')     
    ax2.set_ylabel('$P_i$')
    ax2.legend(strs, ncol = 4, loc = 'upper right')
    ax2.grid()
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,300)
    ax2.legend(strs, ncol = 3, loc = 'upper right')

    strs = ["$P_{%.d}$" % (float(x)) for x in range(3)]
    ax2 = plt.subplot(313)
    for i in range(3):
        ax2.plot(t, Probability3000.T[:,i], c = color[i], lw =1.5)
        ax2.plot(model4.time_control, model4.x[:,i], c = 'black', lw =0.6)
    ax2.legend(strs, ncol = 4, loc = 'upper right')     
    ax2.set_ylabel('$P_i$')
    ax2.legend(strs, ncol = 4, loc = 'upper right')
    ax2.set_xlabel('$t$ [min]')
    ax2.grid()
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,300)
    ax2.legend(strs, ncol = 3, loc = 'upper right')
    
    state_no = np.linspace(0,11,12)
    state_no = np.tile(state_no,[Probability1000.shape[1],1])
    Nws_x = np.sum(model4.x*state_no,axis =1)

    
    absolute_error200 = np.mean(np.abs(Nws200 - Nws_x)[1:19000])
    absolute_error1000 = np.mean(np.abs(Nws1000 - Nws_x)[1:19000])
    absolute_error3000 = np.mean(np.abs(Nws3000 - Nws_x)[1:19000])
    absolute_error5000 = np.mean(np.abs(Nws5000 - Nws_x)[1:19000])
    absolute_error10000 = np.mean(np.abs(Nws10000 - Nws_x)[1:19000])
#    
    relative_error200 = np.mean(np.abs(Nws200 - Nws_x)[1:19000]/Nws_x[1:19000])
    relative_error1000 = np.average(np.abs(Nws1000 - Nws_x)[1:19000]/Nws_x[1:19000])
    relative_error3000 = np.average(np.abs(Nws3000 - Nws_x)[1:19000]/Nws_x[1:19000])
    relative_error5000 = np.average(np.abs(Nws5000 - Nws_x)[1:19000]/Nws_x[1:19000])
    relative_error10000 = np.average(np.abs(Nws10000 - Nws_x)[1:19000]/Nws_x[1:19000])

    
    
    error_estimation_200 = np.mean(np.std(Nws200f[:,:19000], axis = 0))/np.sqrt(200)
    error_estimation_1000 = np.mean(np.std(Nws1000f[:,:19000], axis = 0))/np.sqrt(1000)
    error_estimation_3000 = np.mean(np.std(Nws3000f[:,:19000], axis = 0))/np.sqrt(3000)
    error_estimation_5000 = np.mean(np.std(Nws5000f[:,:19000], axis = 0))/np.sqrt(5000)
    error_estimation_10000 = np.mean(np.std(Nws10000f[:,:19000], axis = 0))/np.sqrt(10000)



        
            
        
