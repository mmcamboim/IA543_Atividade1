# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 10:00:25 2021

@author: mcamboim
"""
import numpy as np

def f(x1,x2):
    return x1**2 + 2*(x2**2) + x1*x2 - 3*x1 - 4*x2

def grad_f(x1,x2):
    nabla_f = np.zeros(2).reshape(2,1)
    nabla_f[0] = 2*x1 + x2 - 3
    nabla_f[1] = 4*x2 + x1 - 4
    return nabla_f