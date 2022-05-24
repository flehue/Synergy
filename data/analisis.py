# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:18:41 2022

@author: flehu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample 


# def Kaplan_Yorke_pato(spec,tol=1e-3):
#     cleanspec = spec
#     cleanspec[np.abs(spec)<tol]=0
#     sumLE = np.cumsum(cleanspec)
    
#     j = np.sum((sumLE>-tol))
#     KYdim = j + sumLE[j-1]/np.abs(cleanspec[j])
#     return KYdim

def Kaplan_Yorke(spec,tol=-3e-4):
    sumita = 0
    i = 0
    while sumita + spec[i] > tol:
        sumita += spec[i]
        i+=1
    return i + 1/abs(spec[i])*sumita


def KY(row):
    raw_spec = [row["LE1"],row["LE2"],row["LE3"],row["LE4"]]
    return Kaplan_Yorke(raw_spec)



filename1 = "2nodes_full_oinfo_table"
data1 = pd.read_csv(filename1+".txt").sort_values(["a","b"])[::-1]
filename2 = "2nodes_spectrum"
data2 = pd.read_csv(filename2+".txt",sep="\t").sort_values(["a","b"])
data2["KY_dim"] = data2.apply(KY,axis=1)

x= data2["KY_dim"]
y = data1["Oinfo"]

plt.scatter(x,y)
# indices = sample(range(12000),120

#%%

a_vals = data1["a"].unique()
b_vals = data1["b"].unique()

var = "Oinfo"
plotmat1 = np.zeros((len(b_vals),len(a_vals)))
for i,a in enumerate(a_vals):
    subdata = data1[data1["a"]==a].sort_values("b")
    column = subdata[var].values
    plotmat1[i,:]=column
plotmat1 = np.flipud(plotmat1)
    
var = "KY_dim"
plotmat2 = np.zeros((len(a_vals),len(b_vals)))
for i,a in enumerate(a_vals):
    subdata = data1[data1["a"]==a].sort_values("b")
    column = subdata[var].values
    plotmat2[i,:]=column
plotmat2 = np.flipud(plotmat2)

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
sns.heatmap(plotmat1)
plt.subplot(1,2,12)
sns.heatmap(plotmat2)
plt.tight_layout()
plt.show()






