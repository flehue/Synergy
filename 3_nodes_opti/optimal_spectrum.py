import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import wico_nodes_maruyama as wc
#import Oinfo
import os
from time import time
import pandas as pd
import importlib
importlib.reload(wc)

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1
outfile = "data/spectrum_optimals_6dic.txt"
dt = 0.001/2
t_trans = 200
t_end = 25000
t_span = np.arange(0,t_end,dt)

optimals = np.loadtxt("data/opti_list_6dic_50tries_pars_oinfo_try.txt")

# plt.figure(figsize=(10,30))
for i in range(optimals.shape[0]):
    row = optimals[i,:]
    if i ==rank: ###cada espectro se corre en un núcleo
        e12,e13,e21,e23,e31,e32,oinfo,trie = row
        trie = int(trie)
        conn = wc.par_to_conn([e12,e13,e21,e23,e31,e32])
        wico_conn = lambda t,X: wc.WiCo(X,t,conn)
        me = "Radau"
        X0 = solve_ivp(wico_conn,(0,t_trans),np.ones(6)/2,method = me,t_eval=(0,t_trans)).y.T[-1,:]
        tick = time()
        spec = wc.L_Spectrum_wico(conn,X0,t_end,dt,d0=1e-6)
        tock = time()
        L1,L2,L3,L4,L5,L6 = np.sort(spec)[::-1]
        print(tock-tick,spec)

        if not os.path.isfile(outfile):
            with open(outfile,'w') as f:
                f.write("try\tLE1\tLE2\tLE3\tLE4\tLE5\tLE6\n")
                
        with open(outfile,'a') as f:
            f.write(f'{trie}\t{L1:1.7f}\t{L2:1.7f}\t{L3:1.7f}\t{L4:1.7f}\t{L5:1.7f}\t{L6:1.7f}\n')
        
        
# spec = edo.L_Spectrum_wico(a,b,X0,t_end,dt,d0=0.00001) ###-8