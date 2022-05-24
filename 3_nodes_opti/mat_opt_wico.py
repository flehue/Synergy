import sys
import numpy as np 
from scipy.integrate import odeint,solve_ivp
import Oinfo
import importlib
import wico_nodes_maruyama as wc 
import emoo
import os
importlib.reload(wc)

this_try = sys.argv[1]

outfile = "data/evols/3nodes_heterogeneous_oinfo_det_6dic_try{}_300g_63nuclei.txt".format(this_try)

# np.random.seed(3)

wc.P_Q = np.array([1.5,0,1.3,0,1.1,0])


variables = [["e12",0,2],
             ["e13",0,2],
             ["e21",0,2],
             ["e23",0,2],
             ["e31",0,2],
             ["e32",0,2]]

objectives = ["Oinfo"]



def func_to_optimize(entries):
    e12,e13,e21,e23,e31,e32 = entries["e12"],entries["e13"],entries["e21"],entries["e23"],entries["e31"],entries["e32"]
    conn = wc.par_to_conn((e12,e13,e21,e23,e31,e32))
    fun = lambda t,X: wc.WiCo(X,t,conn)

    downstep = 10
    rate = 1e3
    dt = 1/rate*downstep
    t_trans = 500
    t_end = 2000
    init = np.random.uniform(size=6)/2
    me = "Radau"   

    t_span= np.arange(0,t_end,dt)
    X0 = solve_ivp(fun,(0,t_trans),init,method = me,t_eval = (0,t_trans)).y.T[-1,:]
    tray = solve_ivp(fun,(0,t_end),X0,method="Radau",t_eval=t_span).y.T
    E = tray[:,::2]

    omegainfo = Oinfo.OInformation(E)
    return {"Oinfo":omegainfo}

def checkpopulation(population,columns,gen):
    best_individual = population[0]
    e12 = best_individual[columns["e12"]]
    e13 = best_individual[columns["e13"]]
    e21 = best_individual[columns["e21"]]
    e23 = best_individual[columns["e23"]]
    e31 = best_individual[columns["e31"]]
    e32 = best_individual[columns["e32"]]
    omegainfo = best_individual[columns["Oinfo"]]

    line = f'{e12:.4f}\t{e13:.4f}\t{e21:.4f}\t{e23:.4f}\t{e31:.4f}\t{e32:.4f}\t{omegainfo:.4f}\n'
    with open(outfile,'a') as f:
        f.write(line)

    print(wc.par_to_conn((e12,e13,e21,e23,e31,e32)), "gen {}, Oinfo = {}".format(gen,omegainfo))

header = f'e12\te13\te21\te23\te31\te32\tOinfo\n'

if not os.path.isfile(outfile):
    with open(outfile,'w') as f:
        f.write(header)

emoo = emoo.Emoo(N = 10, C = 63, variables = variables, objectives = objectives)
# Parameters:
# N: size of population
# C: size of capacity 

emoo.setup(eta_m_0 = 20, eta_c_0 = 20, p_m = 0.2)
# Parameters:
# eta_m_0, eta_c_0: defines the initial strength of the mution and crossover parameter (large values mean weak effect)
# p_m: probabily of mutation of a parameter (holds for each parameter independently)

emoo.get_objectives_error = func_to_optimize
emoo.checkpopulation = checkpopulation

emoo.evolution(generations = 300)





        