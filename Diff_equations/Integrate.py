# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:18:23 2019

@author: Andri
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
#from scipy.optimize import differential_evolution
#from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class Maximum_principle():
    
    """ Cost function of waiting in line """
    @staticmethod
    def g_fun(fun,broj_mesta,price_minute):
        g = np.asarray([fun(i,price_minute) for i in range(broj_mesta)])
        return g
    
    """ Hamiltonian function """
    @staticmethod
    def Ham_fun(c_var, g, x, lamb_val, mu_val,Cc):             
        ham_function = Cc*c_var + np.dot(g,x)
        return ham_function
    
    """ Derivative Hamiltonian function """
    @staticmethod
    def jac_Ham_fun(c_var,g,x,p_var,lamb_val,mu_val,broj_mesta,Cc):
        dQdc = Maximum_principle.dtransdc(broj_mesta,mu_val,lamb_val,c_var)
        derivative = Cc + p_var.T.dot(dQdc.T).dot(x)
        return derivative
                                
    """ Transition matrix Q """
    @staticmethod
    def trans_matrix(broj_mesta,mu_val,lamb_val,c_var):
        lamb_val = lamb_val/c_var
        vec = [mu_val, -(lamb_val+mu_val), lamb_val]
        Qstart = np.zeros(broj_mesta)
        Qend = np.zeros(broj_mesta)
        Qstart[:2] = [-lamb_val, lamb_val]
        Qend[-2:] = [mu_val, -mu_val]
        Q = Maximum_principle.sliding_windows(vec, broj_mesta-2)
        Q = np.vstack((Qstart,Q,Qend))
        return Q

    """ Derivative of transition matrix Q """
    @staticmethod
    def dtransdc(broj_mesta,mu_val,lamb_val,c_var):
        dldc_var = lamb_val/c_var**2
        vec = [0, dldc_var, -dldc_var]
        dQdcstart = np.zeros(broj_mesta)
        dQdcend = np.zeros(broj_mesta)
        dQdcstart[:2] = [dldc_var, -dldc_var]
        dQdc = Maximum_principle.sliding_windows(vec, broj_mesta-2)
        dQdc = np.vstack((dQdcstart,dQdc,dQdcend))
        return dQdc
    
    """ Interpolate between lambdas """
    @staticmethod
    def Lambda_value(t,vreme,Lambda):
        lambda_val = np.interp(t,vreme,Lambda)
        return lambda_val
    
    """ Interpolate between c """
    @staticmethod
    def C_value(t,vreme,c):
        c_val = np.round(np.interp(t,vreme,c))
        return c_val
    
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
        
    def __init__(self, broj_mesta, Lambda, vreme, time_control, Cc, mu_value, price_minute, c_var_pos):
        self.broj_mesta = broj_mesta
        self.Lambda = Lambda
        self.vreme = vreme
        self.Cc = Cc
        self.time_control = time_control
        self.mu_val = mu_value
        self.price_minute = price_minute
        self.c_var_pos = c_var_pos
        
    def ode45_step(f, x, t, dt, *args):
        
        """
        One step of 4th Order Runge-Kutta method
        """
        k = dt
        k1 = dt * f(t, x, *args)
        k2 = dt * f(t + 0.5*k, x + 0.5*k1, *args)
        k3 = dt * f(t + 0.5*k, x + 0.5*k2, *args)
        k4 = dt * f(t + dt, x + k3, *args)
        return x + 1/6. * (k1 + 2*k2 + 2*k3 + k4)
    
    def Optimal_control_round(self,var_init, time_init,fun):
        
        
        """ Objective function """
        def Obj_function(c_var, g, x, lamb_val, mu_val,Cc):             
            objective = Cc*c_var + np.dot(g,x) 
            return objective
        
        
        def ode45(f, t, x0, vreme, Lambda, mu_val,broj_mesta, g_function, Cc, c_var_pos):
            """
            4th Order Runge-Kutta method
            """
            n = len(t)
            x = np.zeros((n, len(x0)))
            c_var = np.zeros(n)
            values = np.zeros(n)
            x[0] = x0
            c_var[0] = 1
            values[0] = Cc*c_var[0]
            for i in range(n-1):
                dt = t[i+1] - t[i]
                lamb_val = Maximum_principle.Lambda_value(t[i],vreme,Lambda)
                Q = [Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var) for c_var in c_var_pos]
                xpom = [Maximum_principle.ode45_step(f, x[i], t[i], dt, Q[j]) for j in range(len(c_var_pos))]                
                obj_fun = [Obj_function(c_var_pos[j],g_function,xpom[j],lamb_val,mu_val,Cc) for j in range(len(c_var_pos))] 
                arg_min = np.argmin(obj_fun)
                c_var[i+1] = c_var_pos[arg_min]
                x[i+1] = xpom[arg_min]
                values[i+1] = Maximum_principle.Ham_fun(c_var[i+1],g_function,x[i+1],lamb_val,mu_val,Cc)
            return x, c_var, values

                
        """ Model for solver """
        def model(t,x,Q):           
            """ Solving diff equation:
                Q is Transtion matrix """                             
            dXdt = np.dot(Q.T,x)
            return dXdt
                      
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)        
        self.x, self.c_var, self.values = ode45(model, self.time_control, var_init, 
                       self.vreme, self.Lambda, self.mu_val, self.broj_mesta, self.g_function,self.Cc,self.c_var_pos)
        
    def Optimal_control_round_obj2(self,var_init, time_init,fun):
        
        
        """ Objective function """
        def Obj_function(c_var, g, x, lamb_val, mu_val,Cc):             
            objective = Cc*c_var + np.dot(g,x) 
            return objective
        
        
        def ode45(f, t, x0, vreme, Lambda, mu_val,broj_mesta, g_function, Cc, c_var_pos):
            """
            4th Order Runge-Kutta method
            """
            n = len(t)
            x = np.zeros((n, len(x0)))
            c_var = np.zeros(n)
            values = np.zeros(n)
            x[0] = x0
            c_var[0] = 1
            values[0] = Cc*c_var[0]
            for i in range(n-1):
                dt = t[i+1] - t[i]
                lamb_val = Maximum_principle.Lambda_value(t[i],vreme,Lambda)
                Q = [Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var) for c_var in c_var_pos]
                xpom = [Maximum_principle.ode45_step(f, x[i], t[i], dt, Q[j]) for j in range(len(c_var_pos))]                
                obj_fun = [Obj_function(c_var_pos[j],g_function,xpom[j],lamb_val,mu_val,Cc) for j in range(len(c_var_pos))] 
                obj_fun = obj_fun + 0.8*np.abs(obj_fun - obj_fun[int(c_var[i]-1)])
                arg_min = np.argmin(obj_fun)
                c_var[i+1] = c_var_pos[arg_min]
                x[i+1] = xpom[arg_min]
                values[i+1] = Maximum_principle.Ham_fun(c_var[i+1],g_function,x[i+1],lamb_val,mu_val,Cc)
            return x, c_var, values

                
        """ Model for solver """
        def model(t,x,Q):           
            """ Solving diff equation:
                Q is Transtion matrix """                             
            dXdt = np.dot(Q.T,x)
            return dXdt
                      
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)        
        self.x_4, self.c_var_4, self.values_4 = ode45(model, self.time_control, var_init, 
                       self.vreme, self.Lambda, self.mu_val, self.broj_mesta, self.g_function,self.Cc,self.c_var_pos)   

        

    def Optimal_control(self,var_init, time_init,fun):
        
        
        """ Objective function """
        def Objective_fun(c_var, g, x, t, dt, lamb_val, mu_val, broj_mesta, Cc):
            Q = Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var)             
            xpom = Maximum_principle.ode45_step(model, x, t, dt, Q) 
            ham_function = Cc*c_var + np.dot(g,xpom)
            return ham_function
         
            
         
        """ 4th Order Runge-Kutta method """
        def ode45(f, t, x0, vreme, Lambda, mu_val,broj_mesta, g_function, Cc, c_var_pos):

            n = len(t)
            x = np.zeros((n, len(x0)))
            c_var = np.zeros(n)
            values = np.zeros(n)
            bnd = ((c_var_pos[0],c_var_pos[-1]),)
            x[0] = x0
            c_var[0] = 1
            values[0] = Cc*c_var[0]
            for i in range(n-1):
                dt = t[i+1] - t[i]
                lamb_val = Maximum_principle.Lambda_value(t[i],vreme,Lambda)                
                res  = minimize(Objective_fun, c_var[i], method='TNC', bounds = bnd,
                                       args = (g_function,x[i],t[i],dt,lamb_val,mu_val,broj_mesta,Cc))
                c_var[i+1] = res.x
                Q = Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var[i+1])
                x[i+1] = Maximum_principle.ode45_step(model,x[i],t[i],dt,Q)
                values[i+1] = Maximum_principle.Ham_fun(c_var[i+1],g_function,x[i+1],lamb_val,mu_val,Cc)
            return x, c_var, values

                
        """ Model for solver """
        def model(t,x,Q):           
            """ Solving diff equation:
                Q is Transtion matrix """                              
            dXdt = np.dot(Q.T,x)
            return dXdt
                      
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)        
        self.x_2, self.c_var_2, self.values_2 = ode45(model, self.time_control, var_init, 
                       self.vreme, self.Lambda, self.mu_val, self.broj_mesta, self.g_function,self.Cc,self.c_var_pos)
        
    
    def Optimal_control_cont(self,var_init, time_init,fun):
        
        def ode45(f, t, x0, vreme, Lambda, mu_val,broj_mesta, g_function, Cc, c_var):
            """
            4th Order Runge-Kutta method
            """
            n = len(t)
            x = np.zeros((n, len(x0)))
            values = np.zeros(n)
            x[0] = x0
            c_var1 = np.round(c_var.copy())
            values[0] = Cc*c_var1[0]
            for i in range(n-1):
                dt = t[i+1] - t[i]
                lamb_val = Maximum_principle.Lambda_value(t[i],vreme,Lambda)
                Q = Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var1[i+1])
                x[i+1] = Maximum_principle.ode45_step(f, x[i], t[i], dt, Q)                
                values[i+1] = Maximum_principle.Ham_fun(c_var1[i+1], g_function, x[i+1], lamb_val, mu_val, Cc) 
            return x, c_var1, values

        """ Model for solver """
        def model(t,x,Q):           
            """ Solving diff equation:
                Q is Transtion matrix """                              
            dXdt = np.dot(Q.T,x)
            return dXdt
                      
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)        
        self.x_3, self.c_var_3, self.values_3 = ode45(model, self.time_control, var_init, 
                       self.vreme, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc, self.c_var_2)
        
    def Evaluate_values(self,var_init,c_var,fun):
        
        def ode45(f, t, x0, vreme, Lambda, mu_val,broj_mesta, g_function, Cc, c_var):
            """
            4th Order Runge-Kutta method
            """
            n = len(t)
            x = np.zeros((n, len(x0)))
            values = np.zeros(n)
            x[0] = x0
            values[0] = Cc*c_var[0]
            for i in range(n-1):
                dt = t[i+1] - t[i]
                lamb_val = Maximum_principle.Lambda_value(t[i],vreme,Lambda)
                Q = Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var[i+1])
                x[i+1] = Maximum_principle.ode45_step(f, x[i], t[i], dt, Q)                
                values[i+1] = Maximum_principle.Ham_fun(c_var[i+1], g_function, x[i+1], lamb_val, mu_val, Cc) 
            return x, values

        """ Model for solver """
        def model(t,x,Q):           
            """ Solving diff equation:
                Q is Transtion matrix """                              
            dXdt = np.dot(Q.T,x)
            return dXdt
                      
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)  
        x, values = ode45(model, self.time_control, var_init, 
                       self.vreme, self.Lambda, self.mu_val, self.broj_mesta, self.g_function, self.Cc, c_var)                        
        return x, values
    
    
    
if __name__ == "__main__":
    """ Data """
    def fun(i,price_minute):
        f = price_minute**i*i
        return f
    
    output = pd.read_csv("Output_ext.csv", index_col = 0)
    Lambda = output['lambda'].values[:200]
    vreme = np.linspace(0,199*5,200)
    time_control = np.linspace(0,400,10000)
    broj_mesta = 30
    mu_val = 4.05103447
    Cc = 30
    price_minute = 3
    c_var_pos = np.arange(1,12)
    
    
    Xinit = np.zeros(broj_mesta)
    t_x_init = np.zeros(broj_mesta)
    Xinit[0] = 1
    var_init = Xinit
    time_init = t_x_init
    
    model1 = Maximum_principle(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_pos)
    model1.Optimal_control_round(var_init,time_init,fun)
    model1.Optimal_control(var_init,time_init,fun)
    model1.Optimal_control_cont(var_init,time_init,fun)
    model1.Optimal_control_round_obj2(var_init,time_init,fun)
    
    
    Total_value_1 = np.trapz(model1.values,time_control)
    Total_value_2 = np.trapz(model1.values_2,time_control)
    Total_value_3 = np.trapz(model1.values_3,time_control)
    Total_value_4 = np.trapz(model1.values_4,time_control)
    
    np.savetxt('c_var.csv',model1.c_var_2)
    
    print("Total_value_1 = {}".format(Total_value_1))
    print("Total_value_2 = {}".format(Total_value_2))
    print("Total_value_3 = {}".format(Total_value_3))
    print("Total_value_4 = {}".format(Total_value_4))
    
    """ Graph drawing """
    strs = ["$P_{%.d}$" % (float(x)) for x in range(broj_mesta)]
    figure = plt.figure(figsize=(13, 9))
    ax1 = plt.subplot(311)
    ax1.plot(model1.time_control,model1.x)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,model1.time_control[-1])
    ax1.set_ylabel('$P_i$')
    ax1.grid()
    ax1.legend(strs, ncol = 3, loc = 'upper right')
    
    ax2 = plt.subplot(312)
    ax2.plot(model1.time_control,model1.c_var, 'black')
    ax2.set_ylim(0,c_var_pos[-1]+1)
    ax2.set_xlim(0,model1.time_control[-1])
    ax2.set_ylabel('$c(t)$')
    ax2.grid()

    
    ax3 = plt.subplot(313)
    ax3.plot(model1.time_control,model1.values)
    ax3.set_ylim(0,np.max(model1.values)+50)
    ax3.set_xlim(0,model1.time_control[-1])
    ax3.set_ylabel('$Value$ [EUR/minute]' )
    ax3.set_xlabel('$t$ [min]')
    ax3.grid()

    
    figure2 = plt.figure(figsize=(13, 9))
    ax4 = plt.subplot(311)
    ax4.plot(model1.time_control,model1.x_2)
    ax4.set_ylim(0,1)
    ax4.set_xlim(0,model1.time_control[-1])
    ax4.set_ylabel('$P_i$')
    ax4.grid()
    ax4.legend(strs, ncol = 3, loc = 'upper right')
    
    ax5 = plt.subplot(312)
    ax5.plot(model1.time_control,model1.c_var_2, 'black')
    ax5.set_ylim(0,c_var_pos[-1]+1)
    ax5.set_xlim(0,model1.time_control[-1])
    ax5.set_ylabel('$c(t)$')
    ax5.grid()

    
    ax6 = plt.subplot(313)
    ax6.plot(model1.time_control,model1.values_2)
    ax6.set_ylim(0,np.max(model1.values_2)+50)
    ax6.set_xlim(0,model1.time_control[-1])
    ax6.set_ylabel('$Value$ [EUR/minute]' )
    ax6.set_xlabel('$t$ [min]')
    ax6.grid()

    
    figure3 = plt.figure(figsize=(13, 9))
    ax4 = plt.subplot(311)
    ax4.plot(model1.time_control,model1.x_3)
    ax4.set_ylim(0,1)
    ax4.set_xlim(0,model1.time_control[-1])
    ax4.set_ylabel('$P_i$')
    ax4.grid()
    ax4.legend(strs, ncol = 3, loc = 'upper right')
    
    ax5 = plt.subplot(312)
    ax5.plot(model1.time_control,model1.c_var_3, 'black')
    ax5.set_ylim(0,c_var_pos[-1]+1)
    ax5.set_xlim(0,model1.time_control[-1])
    ax5.set_ylabel('$c(t)$')
    ax5.grid()

    
    ax6 = plt.subplot(313)
    ax6.plot(model1.time_control,model1.values_3)
    ax6.set_ylim(0,np.max(model1.values_3)+50)
    ax6.set_xlim(0,model1.time_control[-1])
    ax6.set_ylabel('$Value(t)$')
    ax6.set_xlabel('$t$ [min]')
    ax6.grid()

    
    figure4 = plt.figure(figsize=(13, 9))
    ax4 = plt.subplot(311)
    ax4.plot(model1.time_control,model1.x_4)
    ax4.set_ylim(0,1)
    ax4.set_xlim(0,model1.time_control[-1])
    ax4.set_ylabel('$P_i$')
    ax4.grid()
    ax4.legend(strs, ncol = 3, loc = 'upper right')
    
    ax5 = plt.subplot(312)
    ax5.plot(model1.time_control,model1.c_var_4, 'black')
    ax5.set_ylim(0,c_var_pos[-1]+1)
    ax5.set_xlim(0,model1.time_control[-1])
    ax5.set_ylabel('$c(t)$')
    ax5.grid()

    
    ax6 = plt.subplot(313)
    ax6.plot(model1.time_control,model1.values_4)
    ax6.set_ylim(0,np.max(model1.values_4)+50)
    ax6.set_xlim(0,model1.time_control[-1])
    ax1.set_ylabel('$Value$ [EUR/minute]' )
    ax6.set_xlabel('$t$ [min]')
    ax6.grid()
