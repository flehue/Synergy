# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:45:35 2022

@author: flehu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wico_nodes_2nodes as wc
from scipy.integrate import solve_ivp

a,b = 0.16,0.005
#a,b = 0.94,0.115 es claramente un ciclo limite pero su KY esta dando > 3


#%%

dt = 0.001
downsamp = 50
t_trans = 500 #en milisegundos
t_end = 500
n_surr = 40 
t_span = np.arange(0,t_end,dt*downsamp)
wico_ab = lambda t,X: wc.WiCo(X,t,a,b)
X0 = solve_ivp(wico_ab,(0,t_trans),np.random.uniform(size=4)/2,t_eval=(t_trans,)).y.T[-1,:]
x_tP= solve_ivp(wico_ab,(0,t_end),X0,t_eval=t_span,method="Radau").y.T

#%%
E1,I1,E2,I2 = [x_tP[:,i] for i in range(4)]


plt.figure(4)
plt.clf()
plt.subplot(1,2,1)
plt.plot(E1,I1)
plt.subplot(1,2,2)
plt.plot(E2,I2)
plt.show()

fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
ax.plot(E1[:-10],E2[:-10],I1[:-10],linewidth=.8)
ax.plot(E1[-10:],E2[-10:],I1[-10:],color="red",linewidth=2)
plt.show()

# %reset -f