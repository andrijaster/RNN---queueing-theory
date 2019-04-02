# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:51:57 2019

@author: Andri
"""

import numpy as np

def batches(x, batchsize):
    for i in range(0, x.shape[0], batchsize):
        yield x[i:i+batchsize]
        
        

x = np.random.rand(50,3)
aa = list(batches(x,8))


