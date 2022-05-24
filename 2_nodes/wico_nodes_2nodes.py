from numba import jit,njit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.linalg import qr,norm 
import Oinfo
import scipy
#import sdeint



connMat=np.array([[16   ,  15 , 5.8 , 0.2 ],  
                  [ -12  ,  -3 ,  0,  0 ],
                  [  1    ,  0.8 ,  16  , 15  ],
                  [  0 ,  0 , -12 , -3. ]])
#Model parameters
P=1*np.array([1.5, 0., 0.2, 0.]) 
tau=np.array([1., 1., 1., 1.])
theta=np.array([4., 3.7, 4., 3.7])
sigma=np.array([1.3, 2., 1.3, 2.])

@njit
def S(x):
    S1=1/(1+np.exp(-sigma*(x-theta)))
    return S1
@njit
def DS(x):
    return sigma*np.exp(-sigma*(x-theta))*(S(x))**2
@njit
def integration(x,conn):
    return x@conn

@njit
def WiCo(x,t,a,b):
    conn=np.array([[16.,15.,5.8,0.2],[-12.,-3.,0.,0.],[a,b,16.,15.],[0.,0.,-12.,-3.]])
    Iinp=S(P + integration(x,conn))
    fun=-x + (1-x)*Iinp
    return fun/tau 
@njit
def WiCo_Jac(x,t,a,b):
    x0,x1,x2,x3 = x
    conn=np.array([[16.,15.,5.8,0.2],[-12.,-3.,0.,0.],[a,b,16.,15.],[0.,0.,-12.,-3.]])
    #jac = -np.eye(4) - np.diag(S(P+ integration(x,conn))) + (1-x)*DS(P + integration(x,conn))*conn
    Sfloat = lambda x,i: S(x)[i]
    DSfloat = lambda x,i: DS(x)[i]
    diag = lambda i: -1 + (1-x[i])*DSfloat(P[i] + x @ conn[:,i],i) * conn[i,i] - Sfloat(P[i] + x@conn[:,i],i)
    nodiag = lambda i,j: (1-x[i])*DSfloat(x@conn[:,i]+P[i],i)*conn[j,i]

    dfdx = np.array([[diag(0),nodiag(0,1),nodiag(0,2),nodiag(0,3)],
                       [nodiag(1,0),diag(1),nodiag(1,2),nodiag(1,3)],
                       [nodiag(2,0),nodiag(2,1),diag(2),nodiag(2,3)],
                       [nodiag(3,0),nodiag(3,1),nodiag(3,2),diag(3)]])
    return dfdx/tau


# def Jac_WiCo(x,t,a,b):
#     conn = np.array([[16.,15.,5.8,0.2],[-12.,-3.,0.,0.],[a,b,16.,15.],[0.,0.,-12.,-3.]])
    

@jit
def Lorenz(X,t,s,b,r):
    x,y,z=X
    return np.array([s*(y-x),x*(r-z)-y,x*y-b*z])

@jit
def Lorenz_Jac(X,t,s,b,r):
    x,y,z = X
    Jac = np.array([[-s,s,0],[r-z,-1,-x],[y,x,-b]])
    return Jac

@jit
def Rossler(X,t,a,b,c):
    x,y,z = X
    return np.array([-y-z,x+a*y,b+z*(x-c)])

def MLE(func,X0,t,dt,d0=0.00001):  ###para comparar
    #d0 = 0.00001
    Pert = np.zeros(X0.shape)
    Pert[0]=d0
    X=X0
    Pert = Pert + X
    T = int(t/dt)#cantidad de puntos a iterar
    sum0=0 #guardamos la sumita
    #M_t = np.zeros(T)
    for i in range(T):
        X += dt*func(X,i*dt)
        Pert += dt*func(Pert,i*dt)
        d1 = np.linalg.norm(Pert-X)
        lambda_i = np.log(d1/d0)/dt
        sum0+=lambda_i
        #M_t[i] = sum0/(i+1)
        Pert = X+(Pert-X)*d0/d1
    mles = sum0/T
    return mles#M_t

def st_MLE(func,X0,t,dt,std,d0=0.00001, same_noise = True):
    shap = np.shape(X0)
    Pert = np.zeros(shap)
    Pert[0]=d0
    X=X0
    Pert = Pert + X
    T = int(t/dt)#cantidad de puntos a iterar
    sum0=0 #guardamos la sumita
    M_t = np.zeros(T)
    sqrtdt = np.sqrt(dt)
    dW = sqrtdt*std
    if not same_noise:
        for i in range(T):
            ran1 = np.random.normal(size=shap)
            ran2 = np.random.normal(size=shap)
            X += dt*func(X,i*dt)+dW*ran1
            Pert += dt*func(Pert,i*dt)+dW*ran2
            d1 = np.linalg.norm(Pert-X)
            lambda_i = np.log(d1/d0)/dt
            sum0+=lambda_i
            M_t[i] = sum0/(i+1)
            Pert = X+(Pert-X)*d0/d1
        mles = sum0/T
        return mles,M_t        
    for i in range(T):
        ran1 = np.random.normal(size=shap)
        ran2 = ran1
        X += dt*func(X,i*dt)+dW*ran1
        Pert += dt*func(Pert,i*dt)+dW*ran2
        d1 = np.linalg.norm(Pert-X)
        lambda_i = np.log(d1/d0)/dt
        sum0+=lambda_i
        M_t[i] = sum0/(i+1)
        Pert = X+(Pert-X)*d0/d1
    mles = sum0/T
    return mles,M_t

@jit(nopython = True)
def st_MLE_lorenz(s,b,r,X0,t,dt,std,d0=0.00001, same_noise = True):
    #func = lambda X,t: WiCo(X,t,a,b)
    Pert = np.zeros(3)
    Pert[0]=d0
    X=X0
    Pert = Pert + X
    T = int(t/dt)#cantidad de puntos a iterar
    sum0=0 #guardamos la sumita
    #M_t = np.zeros(T)
    sqrtdt = np.sqrt(dt)
    dW = sqrtdt*std
    if not same_noise:
        for i in range(T):
            ran1 = np.array([np.random.normal(),np.random.normal(),np.random.normal()])
            ran2 = np.array([np.random.normal(),np.random.normal(),np.random.normal()])
            X += dt*Lorenz(X,i*dt,s,b,r)+dW*ran1
            Pert += dt*Lorenz(Pert,i*dt,s,b,r)+dW*ran2
            d1 = np.linalg.norm(Pert-X)
            lambda_i = np.log(d1/d0)/dt
            sum0+=lambda_i
            #M_t[i] = sum0/(i+1)
            Pert = X+(Pert-X)*d0/d1
        mles = sum0/T
        return mles#,M_t        
    for i in range(T):
        ran1 = np.array([np.random.normal(),np.random.normal(),np.random.normal()])
        ran2 = ran1
        X += dt*Lorenz(X,i*dt,s,b,r)+dW*ran1
        Pert += dt*Lorenz(Pert,i*dt,s,b,r)+dW*ran2
        d1 = np.linalg.norm(Pert-X)
        lambda_i = np.log(d1/d0)/dt
        sum0+=lambda_i
        #M_t[i] = sum0/(i+1)
        Pert = X+(Pert-X)*d0/d1
    mles = sum0/T
    return mles#,M_t

    

    
@jit(nopython=True)    
def st_MLE_wico(a,b,X0,t,dt,std,d0=0.00001, same_noise = True):
    Pert = np.zeros(4)
    Pert[0]=d0
    X=X0
    Pert = Pert + X
    T = int(t/dt)#cantidad de puntos a iterar
    sum0=0 #guardamos la sumita
    #M_t = np.zeros(T)
    sqrtdt = np.sqrt(dt)
    dW = sqrtdt*std
    if not same_noise:
        for i in range(T):
            ran1 = np.array([np.random.normal(),np.random.normal(),np.random.normal(),np.random.normal()])
            ran2 = np.array([np.random.normal(),np.random.normal(),np.random.normal(),np.random.normal()])
            X += dt*WiCo(X,i*dt,a,b)+dW*ran1
            Pert += dt*WiCo(Pert,i*dt,a,b)+dW*ran2
            d1 = np.linalg.norm(Pert-X)
            lambda_i = np.log(d1/d0)/dt
            sum0+=lambda_i
            #M_t[i] = sum0/(i+1)
            Pert = X+(Pert-X)*d0/d1
        mles = sum0/T
        return mles#,M_t        
    for i in range(T):
        ran1 = np.array([np.random.normal(),np.random.normal(),np.random.normal(),np.random.normal()])
        ran2 = ran1
        X += dt*WiCo(X,i*dt,a,b)+dW*ran1
        Pert += dt*WiCo(Pert,i*dt,a,b)+dW*ran2
        d1 = np.linalg.norm(Pert-X)
        lambda_i = np.log(d1/d0)/dt
        sum0+=lambda_i
        #M_t[i] = sum0/(i+1)
        Pert = X+(Pert-X)*d0/d1
    mles = sum0/T
    return mles#,M_t
@jit
def st_MLE_Rossler(a,b,c,X0,t,dt,std,d0=0.00001,same_noise = True):
    Pert = np.zeros(3)
    Pert[0]=d0
    X=X0
    Pert = Pert + X
    T = int(t/dt)#cantidad de puntos a iterar
    sum0=0 #guardamos la sumita
    M_t = np.zeros(T)
    sqrtdt = np.sqrt(dt)
    dW = sqrtdt*std
    if not same_noise:
        for i in range(T):
            ran1 = np.array([np.random.normal(),np.random.normal(),np.random.normal()])
            ran2 = np.array([np.random.normal(),np.random.normal(),np.random.normal()])
            X += dt*Rossler(X,i*dt,a,b,c)+dW*ran1
            Pert += dt*Rossler(Pert,i*dt,a,b,c)+dW*ran2
            d1 = np.linalg.norm(Pert-X)
            lambda_i = np.log(d1/d0)/dt
            sum0+=lambda_i
            #M_t[i] = sum0/(i+1)
            Pert = X+(Pert-X)*d0/d1
        mles = sum0/T
        return mles,M_t        
    for i in range(T):
        ran1 = np.array([np.random.normal(),np.random.normal(),np.random.normal()])
        ran2 = ran1
        X += dt*Rossler(X,i*dt,a,b,c)+dW*ran1
        Pert += dt*Rossler(Pert,i*dt,a,b,c)+dW*ran2
        d1 = np.linalg.norm(Pert-X)
        lambda_i = np.log(d1/d0)/dt
        sum0+=lambda_i
        M_t[i] = sum0/(i+1)
        Pert = X+(Pert-X)*d0/d1
    mles = sum0/T
    return mles,M_t

#edo.st_MLE_lorenz(s,b,r,X0,tend,dt,0)
@njit
def L_Spectrum_wico(a,b,X0,t,dt,d0=0.0001):
    X= X0
    T = int(t//dt)
    #Spec_T = np.zeros((T,len(X0)))
    Q = np.eye(len(X0))
    summ = np.zeros(len(X0))
    #conv = np.zeros((T,len(X0)))
    for i in range(T):
        B=Q
        X+=dt*WiCo(X,i*dt,a,b)
        B += dt*np.dot(WiCo_Jac(X,i*dt,a,b),B)
        Q,R = qr(B)
        lambdas = np.log(np.abs(np.diag(R)))
        summ += lambdas
    spec = summ/T/dt
    return spec

def shift(serie,lag):
    return np.concatenate((serie[lag:],serie[:lag]))

def shifted(X):
    lenny,n = X.shape[0],X.shape[1]
    surr = np.zeros_like(X)
    for i in range(n):
        entry_i = X[:,i]
        lag_i = np.random.choice(list(range(lenny)))
        shifted_i = np.concatenate((X[lag_i:,i],X[:lag_i,i]))
        surr[:,i] = shifted_i
    return surr

def ShiftSurr_Oinfo(serie):
    length = serie.shape[1]
    surr_sample = np.zeros_like(serie)
    for j in range(length):
        x_j = serie[:,j]
        random_lag_j = np.random.choice(range(length)).astype(int)
        surr_j = shift(x_j,random_lag_j)
        surr_sample[:,j] = surr_j
    syn = Oinfo.OInformation(surr_sample)
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
    return oi,mean,std,z



def plotear(ar1,ar2,ar3, figsize = (20,10), title = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    if title is not None:
        ax.set_title(title)
    ax.plot(ar1,ar2,ar3)
    plt.show()

# def slope_MLE_st(func,X0,t,dt,std,n,same_noise = False): 
#     d0 = 0.00001
#     shap = X0.shape
#     Pert = np.zeros(shap)
#     Pert[0]=d0
#     X=X0
#     Pert = Pert + X
#     T = int(t/dt)#cantidad de puntos a iterar
#     sum0=np.zeros(n) #guardamos la sumita
#     L_t = np.zeros((T//n,n))
#     if same_noise:
#         for i in range(T):
#             j = i%n #little counter
#             k = i//n #L_t fills n times slower
#             ran1 = np.random.normal(size=shap)
#             ran2 = ran1
#             X += dt*func(X,i*dt)+std*ran1
#             Pert += dt*func(Pert,i*dt)+std*ran2
#             d1 = np.linalg.norm(Pert-X)
#             lambda_j = np.log(d1/d0)/dt
#             sum0[j] += lambda_j #se llena n veces mas lento
#             if i%n==(n-1):
#                 Pert = X+(Pert-X)*d0/d1
#                 L_t[k] = sum0/(k+1)
#             lambdas= sum0/(T//n)
#         return lambdas,L_t
#     else:
#         for i in range(T):
#             j = i%n #little counter
#             k = i//n #L_t fills n times slower
#             ran1 = np.random.normal(size=shap)
#             ran2 = np.random.normal(size=shap)
#             X += dt*func(X,i*dt)+std*ran1
#             Pert += dt*func(Pert,i*dt)+std*ran2
#             d1 = np.linalg.norm(Pert-X)
#             lambda_j = np.log(d1/d0)/dt
#             sum0[j] += lambda_j #se llena n veces mas lento
#             if i%n==(n-1):
#                 Pert = X+(Pert-X)*d0/d1
#                 L_t[k] = sum0/(k+1)
#             lambdas= sum0/(T//n)
#         return lambdas,L_t


# def st_MLE_sdeint(func,X0,t,dt,std,d0=0.00001):  ###para comparar
#     #d0 = 0.00001
#     noisy = lambda X,t: np.diag(std*np.ones(X0.shape))
#     shap = X0.shape
#     Pert = np.zeros(shap)
#     Pert[0]=d0
#     X=X0
#     Pert = Pert + X
#     T = int(t/dt)#cantidad de puntos a iterar
#     sum0=0 #guardamos la sumita
#     M_t = np.zeros(T)
#     for i in range(T):
#         dt_array = np.array([0,dt])
#         X = sdeint.itoint(func, noisy, X,  dt_array)[-1,:]
#         Pert = sdeint.itoint(func, noisy, Pert, dt_array)[-1,:]
#         d1 = np.linalg.norm(Pert-X)
#         lambda_i = np.log(d1/d0)/dt
#         sum0+=lambda_i
#         M_t[i] = sum0/(i+1)
#         Pert = X+(Pert-X)*d0/d1
#     mles = sum0/T
#     return mles,M_t




    
    
    