import numpy as np 
import wico_nodes
import Oinfo
from scipy.integrate import odeint
import os

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1
np.random.seed(rank)

##se corre en 100 núcleos, cada uno corre 13*40= 520 veces

outfile= "data/surrogates_12to12_seed5.txt"

evol = np.loadtxt("data/12conn_synergy_optimization_seed5.txt",skiprows=1)
pars = evol[-1,:12] ##ultima matriz 

conn = wico_nodes.par12_to_conn12(pars)

t_end=200
dt = 0.001
#downsample=250

wico = lambda X,t : wico_nodes.WiCo(X,t,conn)
X0 = odeint(wico,np.ones(6)/2, np.arange(0,200,0.01))[-1,:]
sample = odeint(wico,X0, np.arange(0,t_end,dt))#[::downsample,:3]

header = f"rank\toinfo_surr\n"
if not os.path.isfile(outfile):
    with open(outfile,'w') as f:
        f.write(header)

omegainfo = wico_nodes.ShiftSurr_Oinfo(sample)

line = f"{rank}\t{omegainfo}\n" 
with open(outfile,'a') as f:
    f.write(line)




    