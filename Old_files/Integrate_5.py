# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:18:23 2019

@author: Andri
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import fsolve
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
    def Ham_fun(c_var,g,x,lamb_val,mu_val,broj_mesta,Cc):
        Q = Maximum_principle.trans_matrix(broj_mesta,mu_val,lamb_val,c_var)
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
    
    def Optimal_control(self,var_init, time_init,fun):
                
        """ Model for solver """
        def model(z,t,vreme,Lambda,mu_val,broj_mesta,g_function,Cc,c_var_pos):           
            """ Solving diff equation:
                lambda_val is lambda in time t, vec is badnwith vector,
                Q is Transtion matrix """    
            x = z[:broj_mesta]
            lamb_val = Maximum_principle.Lambda_value(t,vreme,Lambda) 
            hamiltonian = [Maximum_principle.Ham_fun(i,g_function,x,lamb_val,mu_val,broj_mesta,Cc) for i in c_var_pos]
            c_var = c_var_pos[np.argmin(hamiltonian)]
            Q = Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var)
            dXdt = np.dot(Q.T,x)
            return dXdt
        
        def evaluate_sol(x,dspan,g_function,Lambda,Cc,broj_mesta,mu_val,c_var_pos):
            c_var = np.zeros(len(dspan))
            for i in range(len(dspan)):
                print(i)
                lamb_val = Maximum_principle.Lambda_value(dspan[i],vreme,Lambda) 
                hamiltonian = [Maximum_principle.Ham_fun(i,g_function,x[i,:],lamb_val,mu_val,broj_mesta,Cc) for i in c_var_pos]
                c_var[i] = c_var_pos[np.argmin(hamiltonian)]
            return c_var
            
        
    
        self.g_function = Maximum_principle.g_fun(fun,self.broj_mesta,self.price_minute)        
        self.x = odeint(model, var_init[:self.broj_mesta], t=self.time_control, 
                               args = (self.vreme, self.Lambda, self.mu_val, self.broj_mesta, self.g_function,self.Cc,self.c_var_pos))
        self.c_var = evaluate_sol(self.x,self.time_control,self.g_function,self.Lambda,self.Cc,self.broj_mesta,self.mu_val,self.c_var_pos)

        

    def Optimal_control_round(self,var_init, time_init):
                
        """ Model 2 for solver """
        def model2(t,z,vreme,Lambda,mu_val,broj_mesta,g_function,vreme_c,c):
    
            """ Solving diff equation:
                lambda_val is lambda in time t, vec is bandwith vector,
                Q is Transtion matrix """    
            x = z[:broj_mesta]
            p = z[broj_mesta:]
            lamb_val = Maximum_principle.Lambda_value(t,vreme,Lambda)
            c_var = Maximum_principle.C_value(t,vreme_c,c)
            Q = Maximum_principle.trans_matrix(broj_mesta, mu_val, lamb_val,c_var)
            dXdt = np.dot(Q.T,x)
            dpdt = -(g_function + np.dot(Q,p))
            dVardt = np.concatenate((dXdt,dpdt),axis = None)
            return dVardt
        
        self.c_var_round = np.round(self.c_var)
        
        diff2 = ode(model2).set_integrator('dopri5')
        diff2.set_initial_value(var_init, time_init)
        diff2.set_f_params(self.vreme, self.Lambda, self.mu_val, self.broj_mesta,
                           self.g_function, self.time, self.c_var_round)
        i=0
        solution2 = []
        for i in range(len(self.time)):
            solution2.append(diff2.integrate(self.time[i]))
            i += 1
        self.solution2 = np.array(solution2)
        self.x_round = self.solution2[:,:self.broj_mesta]

        
    def evaluate_f_max(self,c_var,x):
        value = [self.Cc*c_var[i] + np.dot(self.g_function,x[i,:]) for i in range(len(self.time_control))]
        return value




""" Data """
def fun(i,price_minute):
    f = price_minute**i*i
    return f

output = pd.read_csv("Output_ext.csv", index_col = 0)
Lambda = output['lambda'].values[:200]
vreme = np.linspace(0,199*5,200)
time_control = np.linspace(0,199*5,1000)
broj_mesta = 30
mu_val = 4.051
Cc = 5
price_minute = 150/100
c_var_pos = np.arange(1,11)


Xinit = np.zeros(broj_mesta)
t_x_init = np.zeros(broj_mesta)
Xinit[5] = 1
var_init = Xinit
time_init = t_x_init

model1 = Maximum_principle(broj_mesta, Lambda, vreme, time_control, Cc, mu_val, price_minute, c_var_pos)
model1.Optimal_control(var_init,time_init,fun)
#model1.Optimal_control_round(var_init,time_init)
vrednost = model1.evaluate_f_max(model1.c_var,model1.x)
#vrednost2 = model1.evaluate_f_max(model1.c_var_round,model1.x_round)



""" Graph drawing """
strs = ["$P_{}$".format(x) for x in range(model1.broj_mesta)]
figure = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(model1.time_control,model1.x)
ax1.set_ylim(0,1)
ax1.set_xlim(0,model1.time_control[-1])
ax1.set_ylabel('$P_i$')
ax1.set_xlabel('$t$ [min]')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax2 = plt.subplot(312)
ax2.plot(model1.time_control,model1.c_var, 'black')
ax2.set_ylim(0,10)
ax2.set_xlim(0,model1.time_control[-1])
ax2.set_ylabel('$c(t)$')
ax2.grid()
ax2.legend('upravljanje',  loc = 'upper right')

ax3 = plt.subplot(313)
ax3.plot(model1.time_control,vrednost, lw = 5)
ax3.set_ylim(0,np.max(vrednost)+50)
ax3.set_xlim(0,model1.time_control[-1])
ax3.set_ylabel('$Valeu(t)$')
ax3.set_xlabel('$t$ [min]')
ax3.grid()
ax3.legend('vrednosti',  loc = 'upper right')

#figure2 = plt.figure(figsize=(13, 9))
#ax4 = plt.subplot(311)
#ax4.plot(model1.time_control,model1.x_round)
#ax4.set_ylim(0,1)
#ax4.set_xlim(0,model1.time_control[-1])
#ax4.set_ylabel('$P_i$')
#ax4.set_xlabel('$t$ [min]')
#ax4.grid()
#ax4.legend(strs, ncol = 3, loc = 'upper right')
#
#ax5 = plt.subplot(312)
#ax5.plot(model1.time_control,model1.c_var_round, 'black')
#ax5.set_ylim(0,10)
#ax5.set_xlim(0,model1.time_control[-1])
#ax5.set_ylabel('$c(t)$')
#ax5.grid()
#ax5.legend('upravljanje',  loc = 'upper right')
#
#ax6 = plt.subplot(313)
#ax6.plot(model1.time_control,vrednost2, lw = 5)
#ax6.set_ylim(0,np.max(vrednost2)+50)
#ax6.set_xlim(0,model1.time[-1])
#ax6.set_ylabel('$Value(t)$')
#ax6.set_xlabel('$t$ [min]')
#ax6.grid()
#ax6.legend('vrednosti',  loc = 'upper right')