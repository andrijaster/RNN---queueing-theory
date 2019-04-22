# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:24:29 2019

@author: Andri
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

""" Cost function of waiting in line """
def g_fun(broj_mesta):
    price_perminute = 40
    g = np.asarray([price_perminute*x**3 for x in range(broj_mesta)])
    return g

""" Hamiltonian function """
def Ham_fun(c_var,g,x,p_var,lamb_val,mu_val,broj_mesta):
    Cc = 70
    Lamb = lamb_val/c_var
    Q = trans_matrix(broj_mesta,mu_val,Lamb)
    ham_function = Cc*c_var + np.dot(g,x) + p_var.T.dot(Q.T).dot(x)
    return ham_function

""" Derivative Hamiltonian function """
def jac_Ham_fun(c_var,g,x,p_var,lamb_val,mu_val,broj_mesta):
    Cc = 70
    dQdc = dtransdc(broj_mesta,mu_val,lamb_val,c_var)
    derivative = Cc + p_var.T.dot(dQdc.T).dot(x)
    return derivative

""" Transition matrix Q """
def trans_matrix(broj_mesta,mu_val,lamb_val):
    vec = [mu_val, -(lamb_val+mu_val), lamb_val]
    Qstart = np.zeros(broj_mesta)
    Qend = np.zeros(broj_mesta)
    Qstart[:2] = [-lamb_val, lamb_val]
    Qend[-2:] = [mu_val, -mu_val]
    Q = sliding_windows(vec, broj_mesta-2)
    Q = np.vstack((Qstart,Q,Qend))
    return Q

def dtransdc(broj_mesta,mu_val,lamb_val,c_var):
    vec = [0, lamb_val/c_var**2, -lamb_val/c_var**2]
    dQdcstart = np.zeros(broj_mesta)
    dQdcend = np.zeros(broj_mesta)
    dQdcstart[:2] = [lamb_val/c_var**2, -lamb_val/c_var**2]
    dQdc = sliding_windows(vec, broj_mesta-2)
    dQdc = np.vstack((dQdcstart,dQdc,dQdcend))
    return dQdc
    
""" Interpolate between lambdas """
def Lambda_value(t,vreme,Lambda):
    lambda_val = np.interp(t,vreme,Lambda)
    return lambda_val

""" Interpolate between c """
def C_value(t,vreme,c):
    c_val = np.round(np.interp(t,vreme,c))
    return c_val

""" Sliding windows for bandwith matrix 
    a is vector len, W is row dimension """
def sliding_windows(a, W):
    a = np.asarray(a)
    p = np.zeros(W-1,dtype=a.dtype)
    b = np.concatenate((p,a,p))
    s = b.strides[0]
    strided = np.lib.stride_tricks.as_strided
    return strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))

""" Model for solver """
def model(t,z,vreme,Lambda,mu_val,broj_mesta,g_function):
    
    """ Solving diff equation:
        lambda_val is lambda in time t, vec is badnwith vector,
        Q is Transtion matrix """    
    x = z[:broj_mesta]
    p = z[broj_mesta:]
    lamb_val = Lambda_value(t,vreme,Lambda)
    bnd = ((1,10),)
    x0 = 3
#    c_var = minimize(Ham_fun,x0,method='TNC', 
#                     args = (g_function,x,p,lamb_val,mu_val,broj_mesta),bounds  
    c_var = minimize(Ham_fun,x0,method='TNC',jac = jac_Ham_fun, 
                 args = (g_function,x,p,lamb_val,mu_val,broj_mesta),bounds = bnd) 
    c_var = c_var.x
    lambda_val = lamb_val/c_var
    Q = trans_matrix(broj_mesta, mu_val, lambda_val)
    dXdt = np.dot(Q.T,x)
    dpdt = -(g_function + np.dot(Q,p))
    dVardt = np.concatenate((dXdt,dpdt),axis = None)
    return dVardt

""" Model 2 for solver """
def model2(t,z,vreme,Lambda,mu_val,broj_mesta,g_function,vreme_c,c):
    
    """ Solving diff equation:
        lambda_val is lambda in time t, vec is bandwith vector,
        Q is Transtion matrix """    
    x = z[:broj_mesta]
    p = z[broj_mesta:]
    lamb_val = Lambda_value(t,vreme,Lambda)
    c_var = C_value(t,vreme_c,c)
    lambda_val = lamb_val/c_var
    Q = trans_matrix(broj_mesta, mu_val, lambda_val)
    dXdt = np.dot(Q.T,x)
    dpdt = -(g_function + np.dot(Q,p))
    dVardt = np.concatenate((dXdt,dpdt),axis = None)
    return dVardt

def evaluate(x,p,Lambda,vreme,t_tot,g_function):   
    c_var = np.zeros(len(t_tot))
    i=0
    for t in t_tot:
        lamb_val = Lambda_value(t,vreme,Lambda)
        bnd = ((1,10),)
        x0 = 3
#        resenje = minimize(Ham_fun,x0,method='TNC', 
#                     args = (g_function,x[i,:],p[i,:],lamb_val,mu_val,broj_mesta),bounds = bnd)
        resenje = minimize(Ham_fun, x0,method='TNC', jac = jac_Ham_fun,
                 args = (g_function,x[i,:],p[i,:],lamb_val,mu_val,broj_mesta),bounds = bnd)
        c_var[i] = resenje.x
        i+=1
    return c_var

def evaluate_f_max(x,t_tot,g_function,c_var):
    i = 0
    vrednost = np.zeros(len(t_tot))
    for t in t_tot:
        Cc = 70
        vrednost[i] = Cc*c_var[i] + np.dot(g_function,x[i,:])
        i +=1
    return vrednost
        
    


""" Data """
output = pd.read_csv("Output_ext.csv", index_col = 0)
Lambda = output['lambda'].values[:200]
vreme = np.linspace(0,199*5,200)

""" M/ M/ c=1/ broj_mesta = m+2 """
broj_mesta = 10
strs = ["$P_{}$".format(x) for x in range(broj_mesta)]

""" Initial conditions """
t1 = 199*5
Xinit = np.zeros(broj_mesta)
t_x_init = np.zeros(broj_mesta)
Xinit[0] = 1
pinit = np.zeros(broj_mesta)
t_p_init = np.ones(broj_mesta)*vreme[-1]
var_init = np.concatenate((Xinit, pinit),axis = None)
time_init = np.concatenate((t_x_init, t_p_init),axis = None)
                     
""" Diff equations calculate c_opt"""                    
mu_val = 4.051
g_function = g_fun(broj_mesta)

X_pocetno = np.random.rand(broj_mesta)
X_pocetno = X_pocetno/sum(X_pocetno)
p_pocetno = np.random.rand(broj_mesta)

t = np.linspace(0,199*5,1000)
diff = ode(model).set_integrator('dopri5')
diff.set_initial_value(var_init, time_init)
diff.set_f_params(vreme, Lambda, mu_val, broj_mesta,g_function)
i=0
solution = []
for i in range(len(t)):
    solution.append(diff.integrate(t[i]))
    i += 1


solution = np.array(solution)

c_var = evaluate(solution[:,:broj_mesta],solution[:,broj_mesta:],Lambda,vreme,t,g_function)
vrednost = evaluate_f_max(solution[:,:broj_mesta],t,g_function,c_var)

""" Round value of c """
vreme_c = t
c_var_round = np.round(c_var)

""" Diff equations calculate for round(c_opt()) """
diff2 = ode(model2).set_integrator('dopri5')
diff2.set_initial_value(var_init, time_init)
diff2.set_f_params(vreme, Lambda, mu_val, broj_mesta,g_function,vreme_c,c_var_round)
i=0
solution2 = []
for i in range(len(t)):
    solution2.append(diff2.integrate(t[i]))
    i += 1
    
solution2 = np.array(solution2)
vrednost2 = evaluate_f_max(solution2[:,:broj_mesta],t,g_function,c_var_round)

""" Graph drawing """
figure = plt.figure(figsize=(13, 9))
ax1 = plt.subplot(311)
ax1.plot(t,solution[:,:broj_mesta])
ax1.set_ylim(0,1)
ax1.set_xlim(0,t[-1])
ax1.set_ylabel('$P_i$')
ax1.set_xlabel('$t$ [min]')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')

ax2 = plt.subplot(312)
ax2.plot(t,c_var, 'black')
ax2.set_ylim(0,10)
ax2.set_xlim(0,t[-1])
ax2.set_ylabel('$c(t)$')
ax2.grid()
ax2.legend('upravljanje',  loc = 'upper right')

ax3 = plt.subplot(313)
ax3.plot(t,vrednost, lw = 5)
ax3.set_ylim(0,np.max(vrednost)+50)
ax3.set_xlim(0,t[-1])
ax3.set_ylabel('$Valeu(t)$')
ax3.set_xlabel('$t$ [min]')
ax3.grid()
ax3.legend('vrednosti',  loc = 'upper right')

figure2 = plt.figure(figsize=(13, 9))
ax4 = plt.subplot(311)
ax4.plot(t,solution2[:,:broj_mesta])
ax4.set_ylim(0,1)
ax4.set_xlim(0,t[-1])
ax4.set_ylabel('$P_i$')
ax4.set_xlabel('$t$ [min]')
ax4.grid()
ax4.legend(strs, ncol = 3, loc = 'upper right')

ax5 = plt.subplot(312)
ax5.plot(t,c_var_round, 'black')
ax5.set_ylim(0,10)
ax5.set_xlim(0,t[-1])
ax5.set_ylabel('$c(t)$')
ax5.grid()
ax5.legend('upravljanje',  loc = 'upper right')

ax6 = plt.subplot(313)
ax6.plot(t,vrednost2, lw = 5)
ax6.set_ylim(0,np.max(vrednost2)+50)
ax6.set_xlim(0,t[-1])
ax6.set_ylabel('$Value(t)$')
ax6.set_xlabel('$t$ [min]')
ax6.grid()
ax6.legend('vrednosti',  loc = 'upper right')