# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:11:27 2019

@author: Andri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')



class Maximum_principle():
    
    """ Cost function of waiting in line """
    @staticmethod
    def g_fun(fun,broj_mesta,price_minute):
        g = np.asarray([fun(i,price_minute) for i in range(broj_mesta)])
        return g  
    
    """ Sliding windows for bandwith matrix 
        a is vector len, W is row dimension """
    @staticmethod
    def sliding_windows(a, W):
        a = np.asarray(a)
        p = np.zeros(W-1,dtype=a.dtype)
        b = np.concatenate((p,a,p))
        s = b.strides[0]
        strided = np.lib.stride_tricks.as_strided
        return strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))
                                
    """ Transition matrix Q """
    @staticmethod
    def trans_matrix(broj_mesta,mu_val,lamb_val):
        vec = [mu_val, -(lamb_val+mu_val), lamb_val]
        Qstart = np.zeros(broj_mesta)
        Qend = np.zeros(broj_mesta)
        Qstart[:2] = [-lamb_val, lamb_val]
        Qend[-2:] = [mu_val, -mu_val]
        Q = Maximum_principle.sliding_windows(vec, broj_mesta-2)
        Q = np.vstack((Qstart,Q,Qend))
        return Q
    
    """ Interpolate between lambdas """
    @staticmethod
    def Lambda_value(t,vreme,Lambda):
        lambda_val = np.interp(t,vreme,Lambda)
        return lambda_val
    

        
    def __init__(self, broj_mesta, Lambda, vreme, time_control, Cc, mu_value, price_minute, c_var_min, c_var_max):
        self.broj_mesta = broj_mesta
        self.Lambda = Lambda
        self.vreme = vreme
        self.Cc = Cc
        self.time_control = time_control
        self.mu_val = mu_value
        self.price_minute = price_minute
        self.c_var_min = c_var_min
        self.c_var_max = c_var_max        
        
    
    def Optimal_control(self, fun, x_init):

        
        
        def dQdc(c_var , t):
            lamb_val = Maximum_principle.Lambda_value(t,self.vreme,self.Lambda)
            vec = [0, lamb_val/c_var**2, -lamb_val/c_var**2]
            dQdcstart = np.zeros(self.broj_mesta)
            dQdcend = np.zeros(self.broj_mesta)
            dQdcstart[:2] = [lamb_val/c_var**2, -lamb_val/c_var**2]
            dQdc = Maximum_principle.sliding_windows(vec, self.broj_mesta-2)
            dQdc = np.vstack((dQdcstart,dQdc,dQdcend))
            return dQdc
        
        def jacobian_fun(c_var, t, x, p):
            jacob = self.Cc + np.dot(self.g_function, x) + np.dot(p.T, dQdc(c_var, t).T).dot(x)
            return jacob
            
        
        """ Hamiltonian """
        def ham_fun(c_var, t, x, p):
            Lamb_val = Maximum_principle.Lambda_value(t,self.vreme,self.Lambda)  
            Lamb_val = Lamb_val/c_var 
            Q_var = Maximum_principle.trans_matrix(self.broj_mesta, self.mu_val, Lamb_val)
            function = c_var*(self.Cc + np.dot(self.g_function,x)) + p.T.dot(Q_var.T).dot(x)
            return function
        
        def ode45_step(f, y, t, dt, *args):
            """
            One step of 4th Order Runge-Kutta method
            """
            k1 = dt * f(t, y, *args)
            k2 = dt * f(t + 0.5*dt, y + 0.5*k1, *args)
            k3 = dt * f(t + 0.5*dt, y + 0.5*k2, *args)
            k4 = dt * f(t + dt, y + k3, *args)
            return y + 1/6. * (k1 + 2*k2 + 2*k3 + k4)
        
        """ Model for solver """
        def model(t, z, Q, c_var):           
            """ Solving diff equation:
                Q is Transtion matrix """   
            x = z[:self.broj_mesta]
            p = z[self.broj_mesta:]                   
            dXdt = np.dot(Q.T,x)
            dpdt = -(c_var*self.g_function + np.dot(Q,p))
            dtotdt = np.concatenate((dXdt,dpdt),axis = None)
            return dtotdt
        
        
        def Obj_function(x_val):
            
            """ Solving diff equation:
                lambda_val is lambda in time t, vec is badnwith vector,
                Q is Transtion matrix """ 
                
            var_init_x = x_val
            var_init_p = np.zeros(self.broj_mesta)
            var_init = np.concatenate((var_init_x, var_init_p),axis = None)
            real_values = np.zeros(self.broj_mesta)
            real_values[0] = 1
            
            t = np.linspace(400,0,1000000)
            n = len(t)
            c_var = np.ones(n)
            n = len(t)
            z = np.zeros((n,len(var_init)))
            z[0,:] = var_init
            for i in range(n-1):
                dt = t[i+1] - t[i]
                x = z[i,:self.broj_mesta]
                p = z[i,self.broj_mesta:]
                bnd_c = ((self.c_var_min, self.c_var_max),)
                c_var_res = optimize.minimize(ham_fun, c_var[i], jac = jacobian_fun, method='TNC', bounds = bnd_c,
                            args = (t[i], x, p))    
                c_var[i] = c_var_res.x  
                lamb_val = Maximum_principle.Lambda_value(t[i], self.vreme, self.Lambda)  
                lamb_val = lamb_val/c_var[i]
                Q = Maximum_principle.trans_matrix(self.broj_mesta, self.mu_val, lamb_val) 
                z[i+1] = ode45_step(model, z[i], t[i], dt, Q, c_var[i])
            Izlaz = Out[:self.broj_mesta]
            Objective = np.linalg.norm(Izlaz - real_values)
            return Objective

        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)   
        bnd = ((0, 1),)*self.broj_mesta
        res  = optimize.minimize(Obj_function, x_init, method='TNC', bounds = bnd)
        self.x_kraj = res.x

if __name__ == "__main__":
    
    def fun(i,price_minute):
        f = price_minute**i*i
        return f
    
    output_ext = pd.read_csv("Output_ext.csv", index_col = 0) 
    Lambda = output_ext.iloc[0,:].values.cumsum()
    Lambda = Lambda[1:]
    vreme = np.linspace(0,400,81)
    time_control = np.linspace(0,400,10000)
    broj_mesta = 8
    mu_val = 4.05103447
    Cc = 10
    price_minute = 5
    c_var_min = 1
    c_var_max = 10
    c_no = 1000
    
    
    Xinit = np.zeros(broj_mesta)
    Xinit[2] = 1
    var_init = Xinit
    
    model1 = Maximum_principle(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max)
    model1.Optimal_control(fun, Xinit)