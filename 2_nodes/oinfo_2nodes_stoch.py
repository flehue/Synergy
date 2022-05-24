import matplotlib
from numba import jit,njit
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp
import edo
import Oinfo
import os
import time
import importlib
importlib.reload(edo)

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

std_used = 0.0001#0.01

outfile = "data/2nodes_Oinfo_rank{}_10dic_std{}_just_to_see_24jan.txt".format(rank,std_used)

dt = 0.001
downsamp = 50
t_trans = 20#200
t_end = 20#200
n_surr = 3#40
t_span = np.arange(0,t_end,dt*downsamp)


inpSD=std_used*np.array([1,1,1,1])
SD_over_sqrtdt=inpSD/np.sqrt(dt)


alpha_vals=np.arange(0,1,0.02/2)
beta_vals=np.arange(0,0.6,0.01/2) 

b = beta_vals[rank]
#MLES = np.zeros(shape=(len(alpha_vals),len(beta_vals)))

ab_vals=[(a,b) for a in alpha_vals for b in beta_vals]

if not os.path.isfile(outfile):
    with open(outfile,'w') as f:
        f.write("a\tb\tH\tOinfo\tOmean\tOstd\tTC\tDTC\tSinfo\n")


for a in alpha_vals:

    
    wico_ab = lambda t,X: edo.WiCo(X,t,a,b)
    
    X0 = odeint(wico_ab,np.ones(4),np.arange(0,t_trans,0.01),tfirst=True)[-1,:]
    x_t= edo.run_stoch(a,b,X0,t_end,dt,SD_over_sqrtdt)
    
    x_t2=x_t[::downsamp,:]
    H=Oinfo.H(x_t2)
    Oinf = Oinfo.OInformation(x_t2)
    surr_vals = np.zeros(n_surr)
    for i in range(n_surr):
        shift_surr= edo.shifted(x_t2)
        shift_Oinf = Oinfo.OInformation(shift_surr)
        surr_vals[i] = shift_Oinf
    Omean,Ostd = surr_vals.mean(),surr_vals.std()
    # Oinf,Omean,Ostd,Oz=edo.Shift_Zscore(x_t,40)
    TC=Oinfo.TC(x_t2)
    Sinfo=2*TC-Oinf
    DTC=TC-Oinf


    with open(outfile,'a') as f:
        f.write(f'{a}\t{b}\t{H}\t{Oinf:1.6f}\t{Omean}\t{Ostd}\t{TC:1.6f}\t{DTC:1.6f}\t{Sinfo:1.6f}\n')
 
