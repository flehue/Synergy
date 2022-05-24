# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:20:58 2022

@author: flehu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
from random import sample
import edo
from scipy.stats import spearmanr,pearsonr

def Kaplan_Yorke(spec,tol=-3e-4):
    sumita = 0
    i = 0
    while sumita + spec[i] > tol:
        sumita += spec[i]
        i+=1
    return i + 1/abs(spec[i])*sumita

def Kaplan_Yorke_pato(spec,tol=1e-3):
    cleanspec = spec
    cleanspec[np.abs(spec)<tol]=0
    sumLE = np.cumsum(cleanspec)
    
    j = np.sum((sumLE>-tol))
    KYdim = j + sumLE[j-1]/np.abs(cleanspec[j])
    return KYdim


def to_float(number):
    return float(f"{number:.3f}")


def color(row):
    valor = row["p_value"]
    if valor<0.05:
        color='tab:blue'
    else:
        color='tab:red'
    return color
# plt.scatter(x=data["KY"],y=data["Oinfo"],color=list(data["signo"]))


data_s = pd.read_csv(f"raw_data_definitive/2nodes_spectrum_fixed_31jan.csv").sort_values(["a","b"]).reset_index(drop=True)
new_a,new_b = data_s.apply(lambda row: to_float(row["a"]),axis=1),data_s.apply(lambda row: to_float(row["b"]),axis=1)
data_s["a"],data_s["b"] = new_a,new_b
# data_s[np.abs(data_s)<1e-4]=0
# data_s["KY_dim"] = data_s.apply(lambda row: Kaplan_Yorke(np.array([row["LE1"],row["LE2"],row["LE3"],row["LE4"]])),axis=1)

data_o = pd.read_csv("raw_data_definitive/2nodes_Oinfo.txt",sep="\t").sort_values(["a","b"]).reset_index(drop=True)
new_a,new_b = data_o.apply(lambda row: to_float(row["a"]),axis=1),data_o.apply(lambda row: to_float(row["b"]),axis=1)
data_o["a"],data_o["b"] = new_a,new_b

new_data = data_s.merge(data_o,on=["a","b"])
new_data["Z"] = (new_data["Oinfo"]-new_data["Omean"])/new_data["Ostd"]

#%%

plt.figure(1)
plt.clf()

plt.subplot(1,2,1)
plt.scatter(new_data["KY_dim"],new_data["Oinfo"])
plt.xlabel("KY dimension")
plt.ylabel("Oinfo")

plt.subplot(1,2,2)
plt.scatter(new_data["LE1"].values,new_data["Oinfo"])
plt.ylabel("Oinfo")
plt.xlabel("MLE")
plt.show()

#%%

index_list = sample(range(12000),1200)

col1,col2 = [],[]

for index in index_list:
    row = new_data.iloc[index]
    KY,Oinfo = row["Oinfo"],row["KY_dim"]
    col1.append(KY)
    col2.append(Oinfo)
    
print(spearmanr(col1,col2))
print(pearsonr(col1,col2))






#%%
var = "Oinfo"
a_vals,b_vals = new_data["a"].unique(),new_data["b"].unique()

plotmat = np.zeros((len(a_vals),len(b_vals)))

for i,a in enumerate(a_vals):
    try:
        subdata = new_data[new_data["a"]==a].sort_values("b")
        column = subdata[var].values
        plotmat[i,:]=column
    except:
        pass
fig = plt.figure(figsize=(10,10))
plt.clf()
sns.heatmap(np.flipud(plotmat))
sns.set(font_scale=1)
plt.xticks(ticks= 2*np.array([0,10,20,30,40,50,59]),labels = [0,0.1,0.2,0.3,0.4,0.5,0.59],fontsize=15)  ##beta
plt.yticks(ticks= 2*np.array([1,11,21,31,41,50]),labels = [0.98,0.8,0.6,0.4,0.2,0],fontsize=15)  ##alfa
plt.xlabel(r"$\beta$",fontsize=15)
plt.ylabel(r"$\alpha$",fontsize=15)
plt.show()

#%%


data_pato = pd.read_csv("raw_data_definitive/pato_noise_1e-4.txt",sep="\t", skiprows=1, names = ["a","b","H","TC","DTC","Oinfo","Sinfo","Omean","Ostd"],index_col=False)
new_a,new_b = data_pato.apply(lambda row: to_float(row["a"]),axis=1),data_pato.apply(lambda row: to_float(row["b"]),axis=1)
data_pato["a"],data_pato["b"] = new_a,new_b


data_pato = data_pato.sort_values("b")
data_pato["Z"] = (data_pato["Oinfo"]-data_pato["Omean"])/data_pato["Ostd"]
a_vals,b_vals = data_pato["a"].unique(),data_pato["b"].unique()

plotmat = np.zeros((50,60))
var = "Omean"


for i,b in enumerate(b_vals):
    subdata = data_pato[data_pato["b"]==b].sort_values("a")[var]
    plotmat[:,i] = subdata
plotmat = np.flipud(plotmat)

plt.figure()
plt.title(var)
sns.heatmap(plotmat,cmap="PuOr",vmin=-3,vmax=3)
plt.show()

#%%

from scipy.stats import ttest_ind

full_thing = data_pato.merge(new_data,on=["a","b"],suffixes=["_pato","_me"])

subset1 = full_thing[full_thing["KY_dim"] == 1]
subset1_mean = subset1["Oinfo_me"].mean()
subset1_std = subset1["Oinfo_me"].std()
print(f"1d, mean = {subset1_mean:.5f}, std ={subset1_std:5f}, size = {len(subset1)}")

subset2 = full_thing[full_thing["KY_dim"] == 2]
subset2_mean = subset2["Oinfo_me"].mean()
subset2_std = subset2["Oinfo_me"].std()
print(f"2d, mean = {subset2_mean:.5f}, std ={subset2_std:5f}, size = {len(subset2)}")

subset3 = full_thing[full_thing["KY_dim"] > 2.2]
subset3_mean = subset3["Oinfo_me"].mean()
subset3_std = subset3["Oinfo_me"].std()
print(f"chaotic, mean = {subset3_mean:.5f}, std ={subset3_std:5f}, size = {len(subset3)}")

plotmat = np.zeros((50,60))
var = "Omean_pato"


for i,b in enumerate(b_vals):
    subdata = full_thing[full_thing["b"]==b].sort_values("a")[var]
    plotmat[:,i] = subdata
plotmat = np.flipud(plotmat)

plt.figure(1)
plt.clf()
plt.title(var)
sns.heatmap(plotmat,cmap="PuOr")#,vmin=-1.5,vmax=1.5)
plt.show()

#%%

t_test,p_value = ttest_ind(subset2["Oinfo_me"],subset3["Oinfo_me"],equal_var=False)
print(t_test,p_value)

#%%% analisis 10 de abril para arreglar grafico de la ZOinfo

# data = pd.read_csv("data/2nodes_Oinfo_rank0_10april_wiconodes_trylonger.txt",sep="\t")

# for i in range(1,120):
#     subdata = pd.read_csv(f"data/2nodes_Oinfo_rank{i}_10april_wiconodes_trylonger.txt",sep="\t")
#     data = data.append(subdata)

data = pd.read_csv("data/2nodes_Oinfo_10april_wiconodes_trylonger.txt")
data["Z"] = (data["Oinfo"]-data["Omean"])/data["Ostd"]
a_vals, b_vals = np.sort(data["a"].unique()), np.sort(data["b"].unique())
plotmat = np.zeros((len(a_vals),len(b_vals)))
var = "Z"


for i,a in enumerate(a_vals):
    subdata = data[data["a"]==a].sort_values("b")
    plotmat[i,:] = subdata[var].values
plotmat = np.flipud(plotmat)
plt.figure(1)
plt.clf()
sns.heatmap(plotmat,cmap="PuOr",vmin=-3,vmax=3)
plt.show()
    


    















