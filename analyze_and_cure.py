# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:53:22 2022

@author: flehu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
from random import sample
from scipy.stats import spearmanr,pearsonr

def Kaplan_Yorke(spec,tol=-3e-4):
    try:
        sumita = 0
        i = 0
        while sumita + spec[i] > tol:
            sumita += spec[i]
            i+=1
        return i + 1/abs(spec[i])*sumita
    except Exception as err:
        print(spec, err)
        return np.nan

def Kaplan_Yorke_pato(spec,tol=1e-3):
    cleanspec = spec
    cleanspec[np.abs(spec)<tol]=0
    sumLE = np.cumsum(cleanspec)
    j = np.sum((sumLE>0))
    KYdim = j + sumLE[j-1]/np.abs(cleanspec[j])
    return KYdim

def fill_plotmat(df,var,shape):
    sorted_thing = df.sort_values(["a","b"])
    plotmat = sorted_thing[var].values.reshape(shape,order="C")
    return np.flipud(plotmat)

def clean_column(column,tol=1e-4):
    vals = np.copy(column.values)
    vals[np.abs(column)<tol] = 0
    return vals

def KY_from_row(row,tol=-3e-4):
    spec = np.array([row["LE1"],row["LE2"],row["LE3"],row["LE4"]])
    if row["LE1"]>0:
        try:
            sumita = 0
            i = 0
            while sumita + spec[i] > tol:
                sumita += spec[i]
                i+=1
            KY_dim = i + 1/abs(spec[i])*sumita
            # if KY_dim>3:
            #     print(row["a"],row["b"],spec,KY_dim)
            return KY_dim
        except Exception as err:
            print(row["a"],row["b"],spec, err)
            return np.nan
    else:
        return (spec==0).sum()
    
    
    

#%%

data_s = pd.read_csv("data/2nodes_spectrum_formatted_18mayo.txt")
data_o = pd.read_csv("data/2nodes_Oinfo_formatted_18mayo.txt")
data = pd.merge(data_s,data_o, on=["a","b"]).sort_values(["a","b"]) #aqui unimos ambos conjuntos de datos
a_vals = data["a"].unique()
b_vals = data["b"].unique()


#%%

def clean_column(column,tol=1e-4):
    vals = np.copy(column.values)
    vals[np.abs(column)<tol] = 0
    return vals

def clean_df(df):
    for i in range(1,5):    
        df[f"LE{i}"] = clean_column(df[f"LE{i}"])
    return df

def getrow(inta,intb,df=data):
    a = a_vals[inta]
    b = b_vals[intb]
    return df[(df["a"]==a) & (df["b"]==b)]

data = clean_df(data)
KYfun = lambda row: KY_from_row(row,tol=0)
data["KY"] = data.apply(KYfun,axis=1)

plotmat = fill_plotmat(data,"KY",(100,120)) 


plt.figure(1)
plt.clf()
plt.imshow(plotmat)#,vmin=-3,vmax=3,cmap="PuOr")
plt.xticks(range(120),range(120))
plt.yticks(range(100),range(100)[::-1])
plt.colorbar()
plt.show()

#notar que el cutoff fue 1e-4, y uno de los valores de parámetros 



"""

ksdafjklsd ya culiao piensa 

por qué a veces lanza 


"""
























# plot1 = fill_plotmat(data,)