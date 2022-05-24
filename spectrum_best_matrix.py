import os
import numpy as np
import gc
import wico_nodes as wc
from scipy.integrate import odeint
from numba import njit
import matplotlib.pyplot as plt
# rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
# threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1


def RK4(f, x, t1, t2, pf, stim=None):
    """
    Fourth-order, 4-step RK routine.
    Returns the step, i.e. approximation to the integral.
    If x is defined at time t_1, then stim should be an array of
    stimulus values at times t_1, (t_1+t_2)/2, and t_2 (i.e. at t1 and t2, as
    well as at the midpoint).
    Alternatively, stim may be a function pointer.
    """
    tmid = (t1 + t2)/2.0
    dt = t2 - t1

    if stim is None:
        pf_in_1 = pf
        pf_in_mid = pf
        pf_in_2 = pf
    else:
        try:
            # test if stim is a function
            s1 = stim(t1)
            s1, smid, s2 = (stim, stim, stim)
        except TypeError:
            #  otherwise assume stim is an array
            s1, smid, s2 = (stim[0], stim[1], stim[2])
        pf_in_1 = (pf, s1)
        pf_in_mid = (pf, smid)
        pf_in_2 = (pf, s2)

    K1 = f(t1, x, pf_in_1)
    K2 = f(tmid, x + dt*K1/2.0, pf_in_mid)
    K3 = f(tmid, x + dt*K2/2.0, pf_in_mid)
    K4 = f(t2, x + dt*K3, pf_in_2)

    return dt * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0

def computeLE(f, fjac, x0, t, p=(), ttrans=None):
    D = len(x0)
    N = len(t)
    if ttrans is not None:
        Ntrans = len(ttrans)
    dt = t[1] - t[0]

    def dPhi_dt(t, Phi, x):
        """ The variational equation """
        D = len(x)
        rPhi = np.reshape(Phi, (D, D))
        rdPhi = np.dot(fjac(t, x, p), rPhi)
        return rdPhi.flatten()

    def dSdt(t, S, p):
        """
        Differential equations for combined state/variational matrix
        propagation. This combined state is called S.
        """
        x = S[:D]
        Phi = S[D:]
        return np.append(f(t,x,p), dPhi_dt(t, Phi, x))

    # integrate transient behavior
    Phi0 = np.eye(D, dtype=np.float64).flatten()
    #S0 = np.append(x0, Phi0)

    if ttrans is not None:
        print("Integrating transient behavior...")
        #Strans = np.zeros((Ntrans, D*(D+1)), dtype=np.float64)
        #Strans[0] = S0
        xi = x0
        for i,(t1,t2) in enumerate(zip(ttrans[:-1], ttrans[1:])):
            xip1 = xi + RK4(f, xi, t1, t2, p)
            #Strans_temp = Strans[i] + RK4(dSdt, Strans[i], t1, t2, p)
            # perform QR decomposition on Phi
            #rPhi = np.reshape(Strans_temp[D:], (D, D))
            #Q,R = np.linalg.qr(rPhi)
            #Strans[i+1] = np.append(Strans_temp[:D], Q.flatten())
            xi = xip1
        x0 = xi

        #S0 = np.append(Strans[-1, :D], Phi0)
        #S0 = Strans[-1]

    # start LE calculation
    LE = np.zeros((D), dtype=np.float64)
    #Ssol = np.zeros((N, D*(D+1)), dtype=np.float64)
    #Ssol[0] = S0
    Ssol_temp = np.append(x0, Phi0)

    print("Integrating system for LE calculation...")
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        Ssol_temp = Ssol_temp + RK4(dSdt, Ssol_temp, t1, t2, p)
        # perform QR decomposition on Phi
        rPhi = np.reshape(Ssol_temp[D:], (D, D))
        Q,R = np.linalg.qr(rPhi)
        Ssol_temp = np.append(Ssol_temp[:D], Q.flatten())
        LE += np.log(np.abs(np.diag(R)))

    print("Computing LE spectrum...")
    LE = LE/N/dt
    return LE

folder = "data"

evol = np.loadtxt(folder + "/evolution_try6_200g.txt",skiprows=1)
pars = evol[-1,:6] ##ultima matriz 
conn = wc.par_to_conn(pars)

fun = lambda t,X,p: wc.WiCo(X,t,conn) ##p es dummy 
jac = lambda t,X,p: wc.WiCo_Jac(X,t,conn)
X0 = odeint(fun,np.ones(6)/2,np.arange(0,500,0.01),tfirst=True,args=(1,))[-1,:]

t = np.arange(0,10000,0.001)
spec = computeLE(fun, jac, X0, t, p=(), ttrans=None)
#print(name,strength,spec)
np.savetxt(folder + "/bestmatrix_evoltry6_spectrum.txt",spec)




# header = "motif\tcoupling\tL1\tL2\tL3\tL4\tL5\tL6\n"
# if not os.path.isfile(folder + f"/motifs_spectrum_9oct.txt"):
#     with open(folder + f"/motifs_spectrum_9oct.txt",'w') as outfile:
#         outfile.write(header)

# line = f"{name}\t{strength:0.2}\t{spec[0]}\t{spec[1]}\t{spec[2]}\t{spec[3]}\t{spec[4]}\t{spec[5]}\n"
# with open(folder + f"/motifs_spectrum_9oct.txt",'a') as f:
#     f.write(line)
# gc.collect()