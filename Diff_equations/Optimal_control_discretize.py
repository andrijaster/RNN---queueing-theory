# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:22:40 2019

@author: Andri
"""

import numpy as np
from scipy import optimize
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')


class Optimal_control_discretize():
    
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
        Q = Optimal_control_discretize.sliding_windows(vec, broj_mesta-2)
        Q = np.vstack((Qstart,Q,Qend))
        return Q
    
    """ Interpolate between lambdas """
    @staticmethod
    def Lambda_value(t,vreme,Lambda):
        lambda_val = np.interp(t,vreme,Lambda)
        return lambda_val
    

        
    def __init__(self, broj_mesta, Lambda, vreme, time_control, Cc, mu_value, price_minute, c_var_min, c_var_max, method = 'TNC'):
        self.broj_mesta = broj_mesta
        self.Lambda = Lambda
        self.vreme = vreme
        self.Cc = Cc
        self.time_control = time_control
        self.mu_val = mu_value
        self.price_minute = price_minute
        self.c_var_min = c_var_min
        self.c_var_max = c_var_max        
        self.methods = method

    
    def Optimal_control(self, var_init, fun, c_no):
        
        """ Interpolate between c """
        def C_value(t,vreme,c):
            c_val = np.interp(t,vreme,c)
            return c_val
        
        def function_evaluate(g_function, x, Cc, c_var):
            function_out = c_var*(Cc + np.dot(g_function,x))
            return function_out
        
        """ Model for solver """
        def model(x, t, vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta):           
            """ Solving diff equation:
                Q is Transtion matrix """ 
            c_var_1 = C_value(t,vreme_c,c_var)     
            lamb_val = Optimal_control_discretize.Lambda_value(t,vreme,Lambda)  
            lamb_val = lamb_val/c_var_1 
            Q = Optimal_control_discretize.trans_matrix(broj_mesta,mu_val,lamb_val)                   
            dXdt = np.dot(Q.T,x)
            return dXdt
        
        
        def Obj_function(c_var, dspan, vreme, vreme_c, Lambda, mu_val, broj_mesta, g_function, Cc):
            
            """ Solving diff equation:
                lambda_val is lambda in time t, vec is badnwith vector,
                Q is Transtion matrix """    
            Out = odeint(model, var_init, dspan,
                       args = (vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta))
            
            function = [function_evaluate(g_function, Out[i,:], Cc, c_var[i]) for i in range(len(c_var))]
            value_fun = np.trapz(function, vreme_c)
            return value_fun
        
        def value(c_var, dspan, vreme, vreme_c, Lambda, mu_val, broj_mesta, g_function, Cc):
            
            Out = odeint(model, var_init, dspan,
                       args = (vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta))
            function = [function_evaluate(g_function, Out[i,:], Cc, c_var[i]) for i in range(len(c_var))]
            return function

            

        self.g_function = Optimal_control_discretize.g_fun(fun,self.broj_mesta,self.price_minute)   
        c_var_initial = np.random.rand(c_no)*(self.c_var_max-1) + self.c_var_min
        vreme_c = np.linspace(self.vreme[0], self.vreme[-1], c_no)
        bnd = ((self.c_var_min, self.c_var_max),)*c_no
        if self.methods == 'TNC':
            res  = optimize.minimize(Obj_function, c_var_initial, method='TNC', bounds = bnd, options={'maxiter':80, 'disp':True},
                            args = (self.time_control, self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc))
        elif self.methods == 'differential_evolution':
            res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = 80, popsize = 1,
                            args = (self.time_control, self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc),  disp = True)
        self.c_var = res.x
        self.values = value(self.c_var, self.time_control,self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc)
        self.tot_value = res.fun
        
        
    def Optimal_control_round(self, var_init, fun, c_no):
        
        """ Interpolate between c """
        def C_value(t,vreme,c_var):
            c_val = c_var[np.argmax(vreme>t)-1]
            return c_val
        
        def function_evaluate(g_function, x, Cc, c_var):
            function_out = c_var*(Cc + np.dot(g_function,x))
            return function_out
        
        """ Model for solver """
        def model(x, t, vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta):           
            """ Solving diff equation:
                Q is Transtion matrix """ 
            c_var_1 = C_value(t,vreme_c,c_var)     
            lamb_val = Optimal_control_discretize.Lambda_value(t,vreme,Lambda)  
            lamb_val = lamb_val/c_var_1 
            Q = Optimal_control_discretize.trans_matrix(broj_mesta,mu_val,lamb_val)                   
            dXdt = np.dot(Q.T,x)
            return dXdt
        
        
        def Obj_function(c_var, dspan, vreme, vreme_c, Lambda, mu_val, broj_mesta, g_function, Cc):
            
            """ Solving diff equation:
                lambda_val is lambda in time t, vec is badnwith vector,
                Q is Transtion matrix """    
            c_var = np.round(c_var)           
            Out = odeint(model, var_init, dspan,
                       args = (vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta))
            
            function = [function_evaluate(g_function, Out[i,:], Cc, c_var[i]) for i in range(len(c_var))]
            value_fun = np.trapz(function, vreme_c)
            return value_fun
        
        def value(c_var, dspan, vreme, vreme_c, Lambda, mu_val, broj_mesta, g_function, Cc):
            
            Out = odeint(model, var_init, dspan,
                       args = (vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta))
            function = [function_evaluate(g_function, Out[i,:], Cc, c_var[i]) for i in range(len(c_var))]
            return function


        self.g_function = Optimal_control_discretize.g_fun(fun,self.broj_mesta,self.price_minute)   
        c_var_initial = np.round(np.ones(c_no)*np.random.rand(c_no)*(self.c_var_max-1) + self.c_var_min)
        vreme_c = np.linspace(self.vreme[0], self.vreme[-1], c_no)
        bnd = ((self.c_var_min, self.c_var_max),)*c_no
        if self.methods == 'TNC':
            res  = optimize.minimize(Obj_function, c_var_initial, method='TNC', bounds = bnd, options={'maxiter':80, 'disp':True},
                            args = (self.time_control, self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc))
        elif self.methods == 'differential_evolution':
            res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = 80, popsize = 1,
                            args = (self.time_control, self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc), disp = True)
        self.c_var_round = np.round(res.x)
        self.values_round = value(self.c_var_round, self.time_control,self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc)
        self.tot_value_round = res.fun        
     
        
    def evaluate_function(self, var_init, fun, c_var_ad, c_no):
        """ Interpolate between c """
        def C_value(t,vreme,c):
            c_val = np.interp(t,vreme,c)
            return c_val
        
        def function_evaluate(g_function, x, Cc, c_var):
            function_out = c_var*(Cc + np.dot(g_function,x))
            return function_out
        
        """ Model for solver """
        def model(x, t, vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta):           
            """ Solving diff equation:
                Q is Transtion matrix """ 
            c_var_1 = C_value(t,vreme_c,c_var)     
            lamb_val = Optimal_control_discretize.Lambda_value(t,vreme,Lambda)  
            lamb_val = lamb_val/c_var_1 
            Q = Optimal_control_discretize.trans_matrix(broj_mesta,mu_val,lamb_val)                   
            dXdt = np.dot(Q.T,x)
            return dXdt
        
        def value(c_var, dspan, vreme, vreme_c, Lambda, mu_val, broj_mesta, g_function, Cc):
            Out = odeint(model, var_init, dspan,
                       args = (vreme_c, vreme, c_var, Lambda, mu_val, broj_mesta))
            function = [function_evaluate(g_function, Out[i,:], Cc, c_var[i]) for i in range(len(c_var))]
            return function, Out
       
        vreme_c = np.linspace(self.vreme[0], self.vreme[-1], c_no)
        
        self.g_function = Optimal_control_discretize.g_fun(fun,self.broj_mesta,self.price_minute)  
        self.values_eva, self.x = value(c_var_ad, self.time_control,self.vreme, vreme_c, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc)
        self.tot_value_eva = np.trapz(self.values_eva, vreme_c)

        
        
        
        
        