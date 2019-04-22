# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:00:21 2019

@author: Andri
"""

import numpy as np
import pandas as pd
from tick.hawkes import SimuInhomogeneousPoisson
from tick.base import TimeFunction
from tick.plot import plot_point_process
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

""" Generate point process """
def generate_times_opt(max_t, delta, vreme_c, c_var, vreme, Lambda):
    time = np.arange(delta,max_t, delta)
    c_value_t = C_value(time,vreme_c,c_var)
    lamb_val_t = Lambda_value(time,vreme,Lambda)/c_value_t
    tf = TimeFunction((time, lamb_val_t), dt=delta)
    Psim = SimuInhomogeneousPoisson([tf], end_time = max_t, verbose = False)
    Psim.simulate()
    simulacija = Psim.timestamps 
    plot_point_process(Psim)
    return simulacija

#class Simulation():
#
#    """ Interpolate between lambdas """    
#    @staticmethod
#    def Lambda_interp(t,vreme,Lambda):
#        lambda_val = np.interp(t,vreme,Lambda)
#        return lambda_val
#    
#    """ Interpolate between C_opt """
#    @staticmethod
#    def C_interp(t, vreme_c, c_opt):
#        c_new = np.round(np.interp(t, vreme_c, c_opt))
#        return c_new    
#    
#    """ Initial condition of system """
#    @staticmethod
#    def initial_sys():
#        t_arrival = 0
#        Waiting = True
#        Nws = 0
#        Nw = 0
#        return t_arrival, Waiting, Nws, Nw
#        
#        
#    def __init__(self,no_sim, tsim, step_size,mu_sim, lambda_sim, c_opt, vreme_c):
#        self.step_size = step_size
#        self.no_sim = no_sim
#        self.tsim = tsim
#        self.mu_sim = mu_sim
#        self.lambda_sim = lambda_sim
#        self.c_opt = c_opt
#        self.vreme_c = vreme_c
#        self.noinst = self.tsim // self.step_size
#        
#    def sim(self):
#        
#        t = np.arange(0, self.tsim, self.step_size)
#        Nws = np.zeros([self.no_sim,self.noinst])
#        NW = np.zeros([self.no_sim,self.noinst])
#        
#        for i in range(self.no_sim):
#            
#            t_arrival = np.zeros(self.noinst)
#            
#            t_arrival, Waiting, Nws[i,j], Nw[i,j] = Simulation.initial_sys()
#            
#            for j in range(0,self.tsim,self.noinst):
#                """ Dolazak jedinice """
#                if t[j] == t_arrival:
#                    if Waiting == True:
#                        tops = generate
#                        tkrit = t[j]+tops
#                        Waiting = False
#                        Nws[i,j] = Nws[j-1]+1
#                    else:
#                        Nws[i,j] = Nws[i,j-1]+1
#                        Nw[i,j] = Nw[i,j-1]+1
#                    t_arrival = t[j] +generate
#                """ Opsuluzivanje jedinice """
#                if t[j] == tkrit:
#                    Nws[j] = Nws[j-1]-1
#                    if Nw[i,j]>0:
#                        tops = generate
#                        tkrit = t+tops
#                        Nw[i,j] = Nw[i,j-1]-1
#                    else:
#                        Waiting = True
#                
                        
                
                        
""" Data """
output = pd.read_csv("Output_ext.csv", index_col = 0)
Lambda = output['lambda'].values[:200]
vreme = np.linspace(0,199*5,200)

#c_var = pd.read_csv("C_varijable.csv", index_col = 0)
c_var = np.ones(199)
vreme_c = np.arange(0,199*5,5)
max_t = 30
delta = 0.001



aaa = generate_times_opt(max_t, delta, vreme_c, c_var, vreme, Lambda)          
            
        
