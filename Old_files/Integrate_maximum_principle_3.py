# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:11:27 2019

@author: Andri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.integrate import solve_bvp
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
        
        """ Model for solver """
        def model(z, t):           
            """ Solving diff equation:
                Q is Transtion matrix """                
            x = z[:self.broj_mesta]
            p = z[self.broj_mesta:]
            c_var_initial = 2
            bnd_c = ((self.c_var_min, self.c_var_max),)
            c_var_res = optimize.minimize(ham_fun, c_var_initial, jac = jacobian_fun, method='TNC', bounds = bnd_c,
                            args = (t, x, p))    
            c_var = c_var_res.x
            lamb_val = Maximum_principle.Lambda_value(t, self.vreme, self.Lambda)  
            lamb_val = lamb_val/c_var 
            Q = Maximum_principle.trans_matrix(self.broj_mesta, self.mu_val, lamb_val)                   
            dXdt = np.dot(Q.T,x)
            dpdt = -(c_var*self.g_function + np.dot(Q,p))
            dtotdt = np.concatenate((dXdt,dpdt),axis = None)
            return dtotdt
        
        """ Model for solver """
        def model_bvp(t, z):           
            """ Solving diff equation:
                Q is Transtion matrix """ 
            dXdt = np.ones((self.broj_mesta,z.shape[1]))
            dpdt = np.ones((self.broj_mesta,z.shape[1]))
            for i in range(z.shape[1]):
                x = z[:self.broj_mesta,i]
                p = z[self.broj_mesta:,i]
                c_var_initial = 2
                bnd_c = ((self.c_var_min, self.c_var_max),)
                c_var_res = optimize.minimize(ham_fun, c_var_initial, jac = jacobian_fun, method='TNC', bounds = bnd_c,
                                args = (t[i], x, p))    
                c_var = c_var_res.x
                lamb_val = Maximum_principle.Lambda_value(t[i], self.vreme, self.Lambda)  
                lamb_val = lamb_val/c_var 
                Q = Maximum_principle.trans_matrix(self.broj_mesta, self.mu_val, lamb_val)                   
                dXdt[:,i] = np.dot(Q.T,x)
                dpdt[:,i] = -(c_var*self.g_function + np.dot(Q,p))
            return np.vstack((dXdt, dpdt))
        
        def bc(y0, y1):
            # Values at t=0:
            x0 = y0[:self.broj_mesta]
            x0[0] = x0[0] -1
            p0 = y0[self.broj_mesta:]
            # Values at t=400:  
            x1 = y1[:self.broj_mesta]
            p1 = y1[self.broj_mesta:]
            # These return values are what we want to be 0:
            return np.hstack([x0, p1])
        
        t_bvp = np.linspace(0,10,100)
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)   
        y_bvp = odeint(model, var_init, t_bvp)
        res  = solve_bvp(lambda t, z: model_bvp(t, z), lambda y0, y1: bc(y0,y1), t_bvp, y_bvp.T)
        return res
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
    Xinit[0] = 1
    p_init = np.random.randn(broj_mesta)
    var_init = np.hstack([Xinit, p_init])
    
    model1 = Maximum_principle(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_min, c_var_max)
    res = model1.Optimal_control(fun, Xinit)