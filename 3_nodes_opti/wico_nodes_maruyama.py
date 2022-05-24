import numpy as np
from scipy import signal
from numba import jit,njit,float64,vectorize
from scipy.integrate import odeint,solve_ivp
from numpy.linalg import qr,norm 
import scipy
import pickle 
import sys
sys.path.append("../")
import Oinfo 


N=6 ##nodos
# D = 2*N ##dimension espacio de fase 
#conexiones excitatorias
a_ee, a_ei =16., 15.
#conexiones inhibitorias
a_ie, a_ii = 12.,3.
#constantes de tiempo
tauE,tauI = 1.,1.
tau = np.array([tauE,tauI,tauE,tauI,tauE,tauI])

#inputs externos
P,Q = 1.5,0.  
P_Q = np.array([1.5,Q,1.3,Q,1.1,Q])
#P_Q = np.array([0.38,Q,0.4,Q,0.43,Q])
#varios:
#rE=rI=0.5
#r = np.array([rE,rI,rE,rI,rE,rI])

mu_E,mu_I=4.,3.7 ###theta en la implementacion edo.py
mu = np.array([mu_E,mu_I,mu_E,mu_I,mu_E,mu_I])

sigma_E,sigma_I=1.3,2.
sigma = np.array([sigma_E,sigma_I,sigma_E,sigma_I,sigma_E,sigma_I])
    

@njit
def S(x):
    S1= 1/(1+np.exp(-sigma*(x-mu)))
    return S1

@njit 
def DS(x):
    return sigma*np.exp(-sigma*(x-mu))*S(x)**2
def par_to_conn(par_list):
    e_12,e_13,e_21,e_23,e_31,e_32 = par_list

    conn = np.array([[a_ee,a_ei,e_12,0.,e_13,0.],
                     [-a_ie,-a_ii,0.,0.,0.,0.],
                     [e_21,0.,a_ee,a_ei,e_23,0.],
                     [0.,0.,-a_ie,-a_ii,0.,0.],
                     [e_31,0.,e_32,0.,a_ee,a_ei],
                     [0.,0.,0.,0.,-a_ie,-a_ii]])
    return conn

    
def par12_to_conn12(par_list):
    e1e2,e1i2 = par_list[:2]
    e1e3,e1i3 = par_list[2:4]
    e2e1,e2i1 = par_list[4:6]
    e2e3,e2i3 = par_list[6:8]
    e3e1,e3i1 = par_list[8:10]
    e3e2,e3i2 = par_list[10:12]
    
    conn = np.array([[a_ee,a_ei,e1e2,e1i2,e1e3,e1i3],
                     [-a_ie,-a_ii,0.,0.,0.,0.],
                     [e2e1,e2i1,a_ee,a_ei,e2e3,e2i3],
                     [0.,0.,-a_ie,-a_ii,0.,0.],
                     [e3e1,e3i1,e3e2,e3i2,a_ee,a_ei],
                     [0.,0.,0.,0.,-a_ie,-a_ii]])
    
    return conn

    
@njit
def WiCo(X,t,conn):
    E,I = X[::2],X[1::2]
    Iinp = S(P_Q + X@conn)
    fun = -X + (1-X)*Iinp
    return fun/tau


@njit
def WiCo_Jac(X,t,conn):
    input_raw = P_Q + X@conn
    Sfloat = lambda x,i: S(x)[i]
    DSfloat = lambda x,i: DS(x)[i]
    diag = lambda i: -1 + (1-X[i])*DSfloat(input_raw[i],i)*conn[i,i]-Sfloat(input_raw[i],i)
    nodiag = lambda i,j: (1-X[i])*DSfloat(input_raw[i],i)*conn[j,i]
    llenar = lambda i,j: diag(i) if i==j else nodiag(i,j)

    dfdx = np.zeros((2*N,2*N)) ##N=3

    for i in range(2*N):
        for j in range(2*N):
            dfdx[i,j] = llenar(i,j)
    return dfdx/tau

@njit
def WiCo_f(S,t,conn): 
    X,Phi = S[:D],S[D:].reshape((D,D))
    dxdt = WiCo(X,t,conn)
    Jac = WiCo_Jac(X,t,conn)
    dPhi_dt = np.dot(Jac, Phi).flatten()
    dSdt = np.append(dxdt,dPhi_dt)
    return dSdt


@njit
def WiCo_stoch(X,t,sqdtD,conn):
    E,I = X[::2],X[1::2]
    Iinp=S(P + X@ conn)
    rand=np.random.normal(0,sqdtD,N)
    fun= (-X + (1-X)*Iinp + rand)/tau
    return fun



@njit
def run_stoch(conn,X0,t,dt,SD_over_sqrtdt):
    time = np.arange(0,t,dt)
    x_t=np.zeros((len(time),N))
    x = X0
    for i,t in enumerate(time):
        x_t[i]=x
        x+=dt*WiCo_stoch(x,t,SD_over_sqrtdt,conn)
    return x_t 

@njit    
def RK4_extended(S,t1,t2,conn): ## S= (X,Phi).flatten()
    tmid = (t1+t2)/2.0
    dt = t2-t1
    
    K1 = WiCo_f(S,t1,conn)
    K2 = WiCo_f(S+dt*K1/2.0,tmid,conn)
    K3 = WiCo_f(S+dt*K2/2.0,tmid,conn)
    K4 = WiCo_f(S+dt*K3,t2,conn)
    
    step = dt*(K1/2.0+K2+K3+K4/2.0)/3.0     
    return step

@njit
def computeLE(conn,X0,t,dt): ############SPECTRUM!!
    tspan = np.arange(0,t,dt)
    T = len(tspan) ##largo de la simulacion

    Phi0 = np.eye(D, dtype=np.float64).flatten()
    LE = np.zeros(D, dtype=np.float64)
    S = np.append(X0, Phi0)
    sumita = np.zeros(D)
    print("Computing LE spectrum...")
    for i,(t1,t2) in enumerate(zip(tspan[:-1], tspan[1:])):
        S += RK4_extended(S,t1, t2,conn)
        # perform QR decomposition on Phi
        x,rPhi = S[:D],np.reshape(S[D:], (D, D))
        Q,R = np.linalg.qr(rPhi)
        S = np.append(x, Q.flatten())
        LE += np.log(np.abs(np.diag(R)))

    # compute LEs
    LE =  LE/T/dt
    return LE

def Kaplan_Yorke(raw_spec,tol=-1e-4):
    spec = np.sort(raw_spec)[::-1]
    sumita = 0
    i = 0
    while sumita + spec[i] > tol:
        sumita += spec[i]
        i+=1
    return i + 1/abs(spec[i])*sumita


def E_synergy(conn,t_end,dt,downstep):
    fun = lambda X,t: WiCo(X,t,conn)
    t_trans = 200
    X0 = odeint(fun,np.ones(2*N)/4,np.arange(0,t_trans,0.01))[-1,:]
    tray = odeint(fun,X0,np.arange(0,t_end,dt))[:,:N]
    sample = tray[::downstep]
    syn = Oinfo.OInformation(sample)
    return syn

def shift(serie,lag):
    return np.concatenate((serie[lag:],serie[:lag]))

def ShiftSurr_Oinfo(serie):
    length = len(serie)
    E1,E2,E3 = [serie[:,i] for i in range(3)]

    random_lag1,random_lag2,random_lag3 = np.random.choice(range(length),size=3).astype(int)
    surr1,surr2,surr3 = shift(E1,random_lag1),shift(E2,random_lag2),shift(E3,random_lag3)

    sample = np.zeros(shape = (length,3))
    sample[:,0],sample[:,1],sample[:,2] = surr1,surr2,surr3
    syn = Oinfo.OInformation(sample)
    return syn

def Shift_Zscore(serie,n_surr):
    oi = Oinfo.OInformation(serie)

    surr_data = []
    for i in range(n_surr):
        surr_oi = ShiftSurr_Oinfo(serie)
        surr_data.append(surr_oi)

    surr = np.array(surr_data)
    mean,std = np.mean(surr),np.std(surr)
    z = (oi-mean)/std 
    p_value = scipy.stats.norm.sf(abs(z))*2
    return oi,z,p_value






@njit 
def WiCo_MLE(conn,X0,t,dt,d0=0.00001):  ###para comparar
    #d0 = 0.00001
    Pert = np.zeros(X0.shape)
    Pert[0]=d0
    X=X0
    Pert = Pert + X
    T = int(t/dt)#cantidad de puntos a iterar
    sum0=0 #guardamos la sumita
    M_t = np.zeros(T)
    for i,te in enumerate(np.arange(0,t,dt)):
        X += dt*WiCo(X,te,conn)
        Pert += dt*WiCo(Pert,te,conn)
        d1 = np.linalg.norm(Pert-X)
        lambda_i = np.log(d1/d0)
        sum0+=lambda_i
        M_t[i] = sum0/(i+1)
        Pert = X+(Pert-X)*d0/d1
    mles = sum0/T/dt 
    return mles#M_t

def WiCo_LSpectrum(conn,X0,t,dt,d0=0.0001):
    X= X0
    #Spec_T = np.zeros((T,len(X0)))
    Q = np.eye(len(X0))
    summ = np.zeros(len(X0))
    tspan = np.arange(0,t,dt)
    T = len(tspan)
    #conv = np.zeros((T,len(X0)))
    for t in tspan:
        B=Q
        X+=dt*WiCo(X,t,conn)
        B += dt*np.dot(WiCo_Jac(X,t,conn),B)
        Q,R = qr(B)
        lambdas = np.log(np.abs(np.diag(R)))
        summ += lambdas
    spec = summ/T/dt
    return spec



def correr(conn,tmax,dt):
    fun = lambda t,X: WiCo(X,t,conn)
    init= np.random.uniform(size=6)
    X0 = solve_ivp(fun,(0,20),init,method = "Radau",t_eval = (0,20)).y.T[-1,:]
    
    tray = solve_ivp(fun,(0,tmax),X0,method = "Radau", t_eval = np.arange(0,tmax,dt)).y.T
    return tray ## E1,I1,E2,I2,E3,I3

def correr_motif(motivo,strength,t,dt):
    with open("../motifs.pickle","rb") as f:
        motifs = pickle.load(f)
    conn = par_to_conn(strength*np.array([motifs[motivo][i,j] for i in range(3) for j in range(3) if i!=j]))
    fun = lambda t,X: WiCo(X,t,conn)
    init= np.random.uniform(size=6)
    print(init)
    X0 = solve_ivp(fun,(0,20),init,method = "Radau",t_eval = (0,20)).y.T[-1,:]
    
    tray = solve_ivp(fun,(0,t),X0,method = "Radau", t_eval = np.arange(0,t,dt)).y.T
    x,y,z = tray[:,0],tray[:,2],tray[:,4]
    return x,y,z


@njit
def L_Spectrum_wico(conn,X0,t,dt,d0=0.0001):
    X= X0
    T = int(t//dt)
    #Spec_T = np.zeros((T,len(X0)))
    Q = np.eye(len(X0))
    summ = np.zeros(len(X0))
    #conv = np.zeros((T,len(X0)))
    for i in range(T):
        B=Q
        X+=dt*WiCo(X,i*dt,conn)
        B += dt*np.dot(WiCo_Jac(X,i*dt,conn),B)
        Q,R = qr(B)
        lambdas = np.log(np.abs(np.diag(R)))
        summ += lambdas
    spec = summ/T/dt
    return spec

def Kaplan_Yorke_pato(spec,tol=1e-3):
    cleanspec = spec
    cleanspec[np.abs(spec)<tol]=0
    sumLE = np.cumsum(cleanspec)
    
    j = np.sum((sumLE>-tol))
    KYdim = j + sumLE[j-1]/np.abs(cleanspec[j])
    return KYdim

#what comes next is from paulzordeba/pyLyapunov en GitHub






    # dfdx = np.array([[diag(0),nodiag(0,1),nodiag(0,2),nodiag(0,3),nodiag(0,4),nodiag(0,5)],
 #                     [nodiag(1,0),diag(1),nodiag(1,2),nodiag(1,3),nodiag(1,4),nodiag(1,5)],
 #                     [nodiag(2,0),nodiag(2,1),diag(2),nodiag(2,3),nodiag(2,4),nodiag(2,5)],
 #                     [nodiag(3,0),nodiag(3,1),nodiag(3,2),diag(3),nodiag(3,4),nodiag(3,5)],
 #                     [nodiag(4,0),nodiag(4,1),nodiag(4,2),nodiag(4,3),diag(4),nodiag(4,5)],
 #                     [nodiag(5,0),nodiag(5,1),nodiag(5,2),nodiag(5,3),nodiag(5,4),diag(4)]])








