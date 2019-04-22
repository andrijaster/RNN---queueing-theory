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
        Nws = np.mean(Nws,axis = 0)
        Nw = np.mean(Nw, axis = 0)
                
        return Nws, Nw, Probability
                
                    
                          
                    
                
            
            
#                
                        
                
                        
""" Data """
output = pd.read_csv("Output_ext.csv", index_col = 0)
Lambda = output['lambda'].values[:200]
vreme = np.linspace(0,199*5,200)

#c_var = pd.read_csv("C_varijable.csv", index_col = 0)
loc_exp = 0.31666666666666665
scale_exp = 0.24685052850561828
c_var = np.loadtxt('c_var.csv')
vreme_c = np.linspace(0,400,10000)
state = 30

model1 = Simulation(10000,400,0.01,loc_exp, scale_exp, Lambda, vreme, c_var, vreme_c, state)
Nws, Nw, Probability = model1.sim()
t = np.arange(Nws.shape[0])


strs = ["$P_{}$".format(x) for x in range(model1.state)]

figure = plt.figure(figsize=(13, 16))
ax1 = plt.subplot(111)
ax1.plot(t,Probability.T)
ax1.set_ylabel('$P_i$')
ax1.set_xlabel('$t$ [min]')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')


        
            
        
