import numpy as np
import pandas as pd
from Oinfo import OInformation as oinfo
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
import os,sys
import wico_nodes as wc

#np.random.seed(13)


rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

outfile = "data/sync_subset_1junio.txt"

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
dt = 0.01
t = np.arange(0,sim_len,dt)
n_iter = 1
n_surr =10

n_sines = 20
n_veces = 40
n_iter = 40

for i in range(n_veces):
    if rank == i:

        oinfos = []

        for n_sync in range(n_sines):

            data = np.zeros((len(t),n_sines))
            sync_freqs = np.random.choice(range(1,40),size = n_sync)
            ugly_freqs = random_primos(n_sines-n_sync)


            for j in range(n_sines):
                if j < n_sync:
                    freq = sync_freqs[j]
                    signal = np.sin(2*np.pi*freq*t)
                    data[:,j] = signal
                else:
                    freq = ugly_freqs[j-n_sync]**.5
                    signal = np.sin(2*np.pi*freq*t)
                    data[:,j]=signal
            oi = oinfo(data)
            oinfos.append(oi) #ya termino de 0 a 19, ahora va todas conmesurables

        data = np.zeros((len(t),n_sines))
        all_sync_freqs = np.random.choice(range(1,40),size = n_sines)
        for j in range(n_sines):
            freq = all_sync_freqs[j]
            signal = np.sin(2*np.pi*freq*t)
            data[:,j] = signal
        oi = oinfo(data)
        oinfos.append(oi)
        

        if not os.path.isfile(outfile):
            with open(outfile,'w') as f:
                f.write("a\tb\tc\tOinfo\tmean\tstd\tzscore\n")

        with open(outfile,'a') as f:
            f.write(f'{a}\t{b}\t{c}\t{oi}\t{mean}\t{std}\t{z_score}\n')

# np.savetxt("sines_data.txt",plotdata)





