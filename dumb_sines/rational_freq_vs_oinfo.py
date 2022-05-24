import numpy as np
import pandas as pd
import os,sys
import scipy
import wico_nodes as wc
from Oinfo import OInformation as Oinfo

outfile = "data/sync_vs_oinfo_7junio_bigdt.txt"
foutfile = "data/sync_vs_oinfo_7junio_freqs_bigdt.txt"

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1
np.random.seed(rank)

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

t_end = 100
dt = .01
t = np.arange(0,t_end,dt)

n_sines = 20
prefreqs = np.array(random_primos(n_sines,low=2,high=100))
freqs = prefreqs**.5
fheader = "rank"
#header para frecuencias
for i in range(n_sines):
    fheader += "\tf{}".format(i+1)
fheader += "\n"
if not os.path.isfile(foutfile):
    with open(foutfile,'w') as f:
        f.write(fheader)
###escribimos las frecuencias
fline = f"{rank}"
for i in range(n_sines):
    fline += f"\t{prefreqs[i]}"
fline += "\n"
with open(foutfile,'a') as f:
    f.write(fline)

    
    


##header para tabla general
header = "rank\tn\tOinfo\tz\tp\n"
if not os.path.isfile(outfile):
    with open(outfile,'w') as f:
        f.write(header)

###todas distintas
data = np.zeros((len(t),n_sines))
for i in range(n_sines):
    signal_i = np.sin(2*np.pi*freqs[i]*t)
    data[:,i] = signal_i
oi,z,p = wc.Shift_Zscore(data,40)

with open(outfile,'a') as f:
    f.write(f'{rank}\t{0}\t{oi}\t{z}\t{p}\n')

###subsets, starting from two until they are all integers
for n_sync in range(2,n_sines+1):
    pretty_freqs = freqs[:n_sync]**2
    ugly_freqs = freqs[n_sync:]
    freqs_to_use = np.concatenate((pretty_freqs,ugly_freqs))
    data = np.zeros((len(t),n_sines))
    for i in range(n_sines):
        signal_i = np.sin(2*np.pi*freqs_to_use[i]*t)
        data[:,i] = signal_i
    oi,z,p = wc.Shift_Zscore(data,40)
    with open(outfile,'a') as f:
        f.write(f'{rank}\t{n_sync}\t{oi}\t{z}\t{p}\n')

    
    

