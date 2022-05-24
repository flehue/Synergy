# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:35:44 2022

@author: flehu
"""

import numpy as np
import matplotlib.pyplot as plt
import wico_nodes_2nodes as wc
from scipy.integrate import solve_ivp
import seaborn as sns
import pandas as pd

def Kaplan_Yorke(spec,tol=-3e-4):
    sumita = 0
    i = 0
    while sumita + spec[i] > tol:
        sumita += spec[i]
        i+=1
    return i + 1/abs(spec[i])*sumita

def Kaplan_Yorke_pato(spec,tol=1e-5):
    cleanspec = spec
    cleanspec[np.abs(spec)<tol]=0
    sumLE = np.cumsum(cleanspec)
    
    j = np.sum((sumLE>-tol))
    KYdim = j + sumLE[j-1]/np.abs(cleanspec[j])
    return KYdim


def KY(row,pato = "True"):
    raw_spec = np.array([row["LE1"],row["LE2"],row["LE3"],row["LE4"]])
    if pato:
        return Kaplan_Yorke_pato(raw_spec)
    else:
        return Kaplan_Yorke(raw_spec)

def to_float(number):
    return float(f"{number:.3f}")

data_s = pd.read_csv("raw_data_definitive/2nodes_spectrum.txt",sep="\t").sort_values(["a","b"])
data_s["KY"] = data_s.apply(KY,axis=1)
var = "KY"
# print(data["KY"])


a_vals,b_vals = data_s["a"].unique(),data_s["b"].unique()
plotmat = np.zeros(shape=(100,120))

for i, a in enumerate(a_vals):
    subdata = data_s[data_s["a"]==a].sort_values("b")
    plotmat[i,:] = subdata[var]
    
    
height,ratio = 10,1.2

fig = plt.figure(1,figsize=(height,height*ratio))
plt.clf()
sns.heatmap(np.flipud(plotmat),cmap = "viridis",vmin=1,vmax=3)
sns.set(font_scale=3)
plt.xticks(ticks= 2*np.array([0,10,20,30,40,50,59]),labels = [0,0.1,0.2,0.3,0.4,0.5,0.59],fontsize=25)  ##beta
plt.yticks(ticks= 2*np.array([1,11,21,31,41,50]),labels = [0.98,0.8,0.6,0.4,0.2,0],fontsize=25)  ##alfa
plt.xlabel(r"$\beta$",fontsize=25)
plt.ylabel(r"$\alpha$",fontsize=25)
plt.tight_layout()
plt.show()

see_data = data_s[data_s["KY"]<0.9]
print(see_data)

#%% visualize shit


dt = 0.001
downsamp = 50
t_trans = 200
t_end = 2000
n_surr = 40
t_span = np.arange(0,t_end,dt*downsamp)

a,b=0,0

wico_ab = lambda t,X: wc.WiCo(X,t,a,b)
    
X0 = solve_ivp(wico_ab,(0,t_trans),np.ones(4),t_eval = (t_trans,)).y.T[-1,:]

x_tP= solve_ivp(wico_ab,(0,t_end),X0,t_eval=t_span,method="Radau")




#%%

data_o = pd.read_csv("data/2nodes_Oinfo_10april_wiconodes_trylonger.txt").sort_values(["a","b"])
data_o["Z"] = (data_o["Oinfo"]-data_o["Omean"])/data_o["Ostd"]
# print(data["KY"])

var = "Z"

a_vals,b_vals = data_o["a"].unique(),data_o["b"].unique()
plotmat = np.zeros(shape=(100,120))

for i, a in enumerate(a_vals):
    subdata = data_o[data_o["a"]==a].sort_values("b")
    plotmat[i,:] = subdata[var]
    
    
height,ratio = 10,1.2

fig = plt.figure(2,figsize=(height,height*ratio))
plt.clf()
sns.heatmap(np.flipud(plotmat),cmap = "PuOr",vmin=-1,vmax=1)
sns.set(font_scale=3)
plt.xticks(ticks= 2*np.array([0,10,20,30,40,50,59]),labels = [0,0.1,0.2,0.3,0.4,0.5,0.59],fontsize=25)  ##beta
plt.yticks(ticks= 2*np.array([1,11,21,31,41,50]),labels = [0.98,0.8,0.6,0.4,0.2,0],fontsize=25)  ##alfa
plt.xlabel(r"$\beta$",fontsize=25)
plt.ylabel(r"$\alpha$",fontsize=25)
plt.tight_layout()
plt.show()

#%%SCATTER

x = np.array(data_s["KY"].values)
y = np.array(data_o["Oinfo"].values)

plt.figure(3)
plt.clf()
plt.scatter(x,y)
plt.show()

