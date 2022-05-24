import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp
# import edo
import Oinfo
import os
import time
import wico_nodes_2nodes as wc

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

outfile = "data/2nodes_Oinfo_rank{}_10april_wiconodes_trylonger.txt".format(rank)

dt = 0.001
downsamp = 50
t_trans = 200
t_end = 2000
n_surr = 40
t_span = np.arange(0,t_end,dt*downsamp)


alpha_vals=np.arange(0,1,0.02/2)
beta_vals=np.arange(0,0.6,0.01/2) 

b = beta_vals[rank]
#MLES = np.zeros(shape=(len(alpha_vals),len(beta_vals)))

ab_vals=[(a,b) for a in alpha_vals for b in beta_vals]

if not os.path.isfile(outfile):
    with open(outfile,'w') as f:
        f.write("a\tb\tH\tOinfo\tOmean\tOstd\tTC\tDTC\tSinfo\n")


for a in alpha_vals:

    
    wico_ab = lambda t,X: wc.WiCo(X,t,a,b)
    
    X0 = odeint(wico_ab,np.ones(4),np.arange(0,t_trans,0.01),tfirst=True)[-1,:]

    x_tP= solve_ivp(wico_ab,(0,t_end),X0,t_eval=t_span,method="Radau")
    
    x_t=x_tP.y.T
    H=Oinfo.H(x_t)
    Oinf = Oinfo.OInformation(x_t)
    surr_vals = np.zeros(n_surr)
    for i in range(n_surr):
        shift_surr= wc.shifted(x_t)
        shift_Oinf = Oinfo.OInformation(shift_surr)
        surr_vals[i] = shift_Oinf
    Omean,Ostd = surr_vals.mean(),surr_vals.std()
    # Oinf,Omean,Ostd,Oz=edo.Shift_Zscore(x_t,40)
    TC=Oinfo.TC(x_t)
    Sinfo=2*TC-Oinf
    DTC=TC-Oinf


    with open(outfile,'a') as f:
        f.write(f'{a}\t{b}\t{H}\t{Oinf:1.6f}\t{Omean}\t{Ostd}\t{TC:1.6f}\t{DTC:1.6f}\t{Sinfo:1.6f}\n')
 
