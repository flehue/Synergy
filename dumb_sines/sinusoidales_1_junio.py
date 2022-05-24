import numpy as np
import pandas as pd
from Oinfo import OInformation as oinfo
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
import os,sys
import wico_nodes as wc



rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

outfile = "data/tres_inconmesurables_1junio.txt"

def is_primo(n):
    if n==2:
        return True
    for i in range(2,int(n**.5)+1):
        if n%i ==0:
            return False
    return True

def random_primos(cuantos,low=2,high=30):
    primos = []
    while len(primos)<cuantos:
        a = np.random.choice(range(low,high))
        if is_primo(a) and a not in primos:
            primos.append(a)
    return primos

sim_len = 100
dt = 0.001

n_iter = 40
n_surr =40

data = np.zeros((n_iter,7))


for i in range(n_iter):
    if i==rank:

        a,b,c = random_primos(3)
        a_root,b_root,c_root = a**.5,b**.5,c
        
        t = np.arange(0,sim_len,dt)
        x = np.sin(2*np.pi*a_root*t)
        y = np.sin(2*np.pi*b_root*t)
        z = np.sin(2*np.pi*c_root*t)

        mat = np.zeros(shape= (len(t),3))
        mat[:,0],mat[:,1],mat[:,2]=x,y,z
        oi = oinfo(mat)
        
        surrogate=np.zeros(n_surr)
        for j in range(n_surr):
            scramble = wc.Surr_Oinfo(mat)
            surrogate[j] = scramble
        mean,std = surrogate.mean(),surrogate.std()
        z_score = (oi-mean)/std
        p_value = scipy.stats.norm.sf(abs(z_score))*2
        print(oi,surrogate.mean(),z_score,p_value)
        
        data[i,:] = a,b,c,oi,surrogate.mean(),z_score,p_value
        if not os.path.isfile(outfile):
            with open(outfile,'w') as f:
                f.write("a\tb\tc\tOinfo\tmean\tstd\tzscore\n")

        with open(outfile,'a') as f:
            f.write(f'{a}\t{b}\t{c}\t{oi}\t{mean}\t{std}\t{z_score}\n')





