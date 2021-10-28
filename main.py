# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 09:28:54 2021

@author: mcamboim
"""
import matplotlib.pyplot as plt
import numpy as np

from rosenbrock_function import grad_f,f

plt.close('all')
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.linewidth'] = 2

# Unidimensional Plot ========================================================
x1_0,x2_0 = (1,2)
nabla_f = grad_f(x1_0,x2_0)

x0 = np.array([x1_0,x2_0]).reshape(2,1)
alpha = np.arange(0,1,0.01)
d_alpha = x0 - nabla_f * alpha
f_alpha = f(d_alpha[0,:],d_alpha[1,:]) 

plt.figure(figsize=(12,6),dpi=150)
plt.plot(alpha,f_alpha,lw=3,c='blue')
plt.xlim([alpha[0],alpha[-1]])
plt.grid(True,ls='dotted')
plt.ylabel('g(\u03B1)')
plt.xlabel('\u03B1')

# Optimum Step ===============================================================
alpha_0 = 0.0
d0 = x0 - nabla_f * alpha_0
nabla_d0 = grad_f(d0[0],d0[1])
lambda_0 = nabla_d0.T @ -nabla_f

alpha_1 = 1.0
d1 = x0 - nabla_f * alpha_1
nabla_d1 = grad_f(d1[0],d1[1])
lambda_1 = nabla_d1.T @ -nabla_f

alpha_opt = (alpha_1 - alpha_0) * lambda_1 / (lambda_0 - lambda_1) + alpha_1 
dopt = x0 - nabla_f * alpha_opt

plt.figure(figsize=(12,6),dpi=150)
plt.plot([alpha_0,alpha_1],[lambda_0[0][0],lambda_1[0][0]],lw=3,c='blue')
plt.xlim([alpha[0],alpha[-1]])
plt.grid(True,ls='dotted')
plt.ylabel('dg(\u03B1)/d\u03B1')
plt.xlabel('\u03B1')

