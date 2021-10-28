# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:26:30 2021

@author: mcamboim
"""

import matplotlib.pyplot as plt
import numpy as np

from rosenbrock_function import grad_f,f

x1 = np.arange(-5,5,0.25)
x2 = np.arange(-5,5,0.25)
x1,x2 = np.meshgrid(x1,x2)
z = f(x1,x2)

fig = plt.figure(figsize=(6,6),dpi=150)
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, z,cmap='viridis', edgecolor='none')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')