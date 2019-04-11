# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:24:29 2019

@author: Andri
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import pandas as pd

def model(t,z,vreme,Lambda,mu_val,broj_mesta):

    """ Interpolate between lambdas """
    def Lambda_value(t,vreme,Lambda):
        lambda_val = np.interp(t,vreme,Lambda)
        return lambda_val

    """ Sliding windows for bandwith matrix 
        a is vector len, W is row dimension """
    
    def sliding_windows(a, W):
        a = np.asarray(a)
        p = np.zeros(W-1,dtype=a.dtype)
        b = np.concatenate((p,a,p))
        s = b.strides[0]
        strided = np.lib.stride_tricks.as_strided
        return strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))
    
    """ Solving diff equation:
        lambda_val is lambda in time t, vec is badnwith vector,
        Q is Transtion matrix """
    
    lamb_val = Lambda_value(t,vreme,Lambda)
    vec = [mu_val, -(lamb_val+mu_val), lamb_val]
    
    Qstart = np.zeros(broj_mesta)
    Qend = np.zeros(broj_mesta)
    Qstart[:2] = [-lamb_val, lamb_val]
    Qend[-2:] = [mu_val, -mu_val]
    Q = sliding_windows(vec, broj_mesta-2)
    Q = np.vstack((Qstart,Q,Qend))
    
    dXdt = np.dot(Q.T,z)
    return dXdt

""" Data """
output = pd.read_csv("Output_ext.csv", index_col = 0)
Lambda = output['lambda'].values[:200]
vreme = np.linspace(0,199*5,200)

""" M/ M/ c=1/ broj_mesta = m+2 """
broj_mesta = 10
strs = ["$P_{}$".format(x) for x in range(broj_mesta)]

""" Diff equations """
t1 = 199*5
p0 = np.zeros(broj_mesta)
t0 = np.zeros(broj_mesta)
p0[0] = 1
mu_val = 4.051
diff = ode(model).set_integrator('dopri5')
diff.set_initial_value(p0,t0)
diff.set_f_params(vreme, Lambda, mu_val, broj_mesta)
t = np.linspace(0,199*5,5000)
i=0
solution = []
for i in range(len(t)):
    solution.append(diff.integrate(t[i]))
    i += 1


figure = plt.figure(figsize=(13, 6))
ax1 = plt.subplot(111)
ax1.plot(t,solution)
ax1.set_ylim(0,1)
ax1.set_xlim(0,1000)
ax1.set_ylabel('$P_i$')
ax1.set_xlabel('$t$ [min]')
ax1.grid()
ax1.legend(strs, ncol = 3, loc = 'upper right')
