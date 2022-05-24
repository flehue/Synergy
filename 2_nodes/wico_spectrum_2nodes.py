import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp
import wico_nodes_2nodes as wc
#import Oinfo
import os
import scipy
import time

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

outfile = "data/2nodes_spectrum_fixed_31jan.txt"

dt = 0.001
t_trans = 200
t_end = 20000
t_span = np.arange(0,t_end,dt)


alpha_vals=np.arange(0,1,0.02/2)  ##se corre en tantos nucleos como betas haya = 120
beta_vals=np.arange(0,0.6,0.01/2) 
b = beta_vals[rank]

#MLES = np.zeros(shape=(len(alpha_vals),len(beta_vals)))

if not os.path.isfile(outfile):
    with open(outfile,'w') as f:
        f.write("a\tb\tLE1\tLE2\tLE3\tLE4\n")

ab_vals=[(a,b) for a in alpha_vals for b in beta_vals]

for a in alpha_vals:

    
    wico_ab = lambda t,X: wc.WiCo(X,t,a,b)
    
    X0 = odeint(wico_ab,np.ones(4),np.arange(0,t_trans,0.01),tfirst=True)[-1,:]

    spec = wc.L_Spectrum_wico(a,b,X0,t_end,dt,d0=0.00001) 
    spec = np.sort(spec)[::-1]
    

    
    with open(outfile,'a') as f:
        f.write(f'{a:0.6f}\t{b:0.6f}\t{spec[0]:1.6f}\t{spec[1]:1.6f}\t{spec[2]:1.6f}\t{spec[3]:1.6f}\n')
    ##ploteamos un par de trayectorias, solo dos dimensiones:
            
        
        
        
        