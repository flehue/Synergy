# -*- coding: utf-8 -*-
"""
Created on Thu May 19 01:26:14 2022

@author: flehu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

##notar que hay un par que entrega KY=4 debido al cutoff elegido, es necesario comentarlo en el paper
#notar que el 19 de mayo que curamos los datos es necesario generar nuevamente el heatmap de KY


#existen parametros que dan KY=1 pero son sinergicos, más aún, los hay caóticos al ver las trayectorias, 
#esto se puede deber a un problema al momento de calcular el threshold de la cosa para el espectro
#de hecho es por esto que sólo se puede confiar en la tendencia estadistica general de los datos

##notar lo que ocurre cuando se consideran solo los valores muy significativos abs(Z)>10

data = pd.read_csv("data/full_data_tol=1e-4_19mayo.txt")
data["Z"] = (data["Oinfo"]-data["Omean"])/data["Ostd"]

def getrow(inta,intb,df=data):
    a = a_vals[inta]
    b = b_vals[intb]
    return df[(df["a"]==a) & (df["b"]==b)]


def getregime(row):
    if row["KY"]==1: #1s
        return 1
    elif (row["KY"]==2) | (row["KY"]==3): #torus
        return 2
    elif ((row["KY"]>2.1) & (row["KY"]<2.9)) | ((row["KY"]>3.1) & (row["KY"]<3.9)): #chaotic
        return 3
    else:
        return 0


subdata = data[(data["KY"]==1) & (data["Oinfo"]<-1)]
signif_data = data[np.abs(data["Z"])>10]

#%% swarmplots dependent on regime

d1_data = data[data["KY"]==1]
meand1 = d1_data["Oinfo"].mean()
stdd1 = d1_data["Oinfo"].std()

torus_data = data[(data["KY"]==2) | (data["KY"]==3)]
meantorus = torus_data["Oinfo"].mean()
stdtorus = torus_data["Oinfo"].std()

chaos_data = data[((data["KY"]>2.1) & (data["KY"]<2.9)) | ((data["KY"]>3.1) & (data["KY"]<3.9))]
meanchaos = chaos_data["Oinfo"].mean()
stdchaos = chaos_data["Oinfo"].std()

data["regime"] = data.apply(getregime,axis=1)

subdf = data.iloc[np.random.choice(range(12000),1000)]

plt.figure(1)
plt.clf()
sns.stripplot(data=subdf, x="regime",y="Oinfo")
plt.tight_layout()
plt.show()








#%%scatter of whole data

x= data["KY"]
y = data["Oinfo"]
plt.figure(2,figsize=(10,5))
plt.clf()
plt.scatter(x,y)
plt.xlabel(r"$\dim_{KY}$",fontsize=30)
plt.ylabel(r"$\Omega$",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.tight_layout()
# plt.savefig("scatter_L1_oinfo.png",dpi=300)
plt.show()

#%% spearman correlation

omega = data["Oinfo"]
KY = data["KY"]
rho, pval = stats.spearmanr(omega, KY)
print(rho,pval) # -0.7463648269713126 0.0

L1 = data["LE1"]
rho, pval = stats.spearmanr(omega, L1)
print(rho,pval) #-0.26015441279192864 7.348483370862279e-185

#%%mean Oinfo

stoch_data = pd.read_csv("data/2nodes_Oinfo_18abril_std0.0001.txt")
print(stoch_data["Oinfo"].mean())







