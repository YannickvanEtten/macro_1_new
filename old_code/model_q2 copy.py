'''
Filename: question.py

Purpose:
    
Date:
    15 August 2023
Author:
    Yannick van Etten 2688877  
'''
###########################################################
### Imports
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.integrate as ing
from scipy.integrate import quad
import pandas as pd
###########################################################
def GetGrid(vRange,iN):
    [dK_min, dK_max] = vRange
    vGrid = np.linspace(dK_min,dK_max,iN)
    return vGrid

def calc_paths(S, K_grid, M_grid, pi, T, a1, a2, b1, b2, phi, alpha, delta, gamma):
    current_index = 0 
    K_path = [K_grid[0]]
    M_path = [M_grid[0]]
    c_path = []
    mu_path = []
    
    for t in range(T):
        next_index = pi[current_index]
        K_t, M_t = S[current_index]
        K_next, M_next = S[next_index]
        
        K_path.append(K_next)
        M_path.append(M_next)
        
        mu = 1- (M_next - (1 - phi) * M_t) / (gamma * K_t**alpha)
        mu_path.append(mu)
        c = (K_t**alpha/(1+b1*M_t**b2))*(1-a1*(mu**a2))+(1-delta)*K_t-K_next
        c_path.append(c)
        
        current_index = next_index
    return K_path, M_path, mu_path, c_path

def steady_state_cap(alpha,beta,delta):
    return(((1 + beta * (delta - 1)) / (beta * alpha))**(1 / (alpha - 1)))

def get_utility(consumption, sigma):
    if (sigma == 1):
        return np.log(consumption)
    return (consumption ** (1 - sigma) - 1) / (1 - sigma)

def iterate2(V_0, pi_0, u_mat, beta, epsilon, i, n):
    V_mat = np.tile(V_0, (n, 1))
    V_1 = np.max(u_mat + beta * V_mat, axis=1) 
    pi = np.argmax(u_mat + beta * V_mat, axis=1)
    if np.linalg.norm(V_1 - V_0) < epsilon:
        return V_1, pi
    print(i, np.linalg.norm(V_1 - V_0))
    i += 1
    return iterate2(V_1, pi_0, u_mat, beta, epsilon, i, n)

def iterate(V_0, pi_0, u_mat, beta, epsilon, i, n):
    V_1 = np.zeros(n) 
    pi = np.zeros(n, dtype=int)
    for k in range(n):
        V_1[k] = np.max(u_mat[k] + beta * V_0)
        pi[k] = np.argmax(u_mat[k] + beta * V_0)
    if np.linalg.norm(V_1 - V_0) < epsilon:
        return V_1, pi
    print(i, np.linalg.norm(V_1 - V_0))
    i += 1
    return iterate(V_1, pi_0, u_mat, beta, epsilon, i, n)

def fUtility(dSigma, dC):
    if dSigma == 1:
      dU = np.log(dC)
    else:
      dU = (dC**(1-dSigma)-1)/(1-dSigma)
    return dU 
def ComputeUmat(vGrid1,vGrid2,dDelta,dAlpha,dBeta,dSigma, vParC):
    iN_1 = len(vGrid1)
    iN_2 = len(vGrid2)
    
    iN = iN_1 * iN_2
    mMesh = np.meshgrid(vGrid1,vGrid2)
    mS = np.array(mMesh).T.reshape(-1,2)
    mUmat = np.zeros((iN,iN))

    for j in range(iN):
        vS_prime = mS[j,:]
        mUmat[:,j] = fUtilde(mS,vS_prime,dDelta,dAlpha,dSigma,vParC)
    return [mS,mUmat]

def fProd(mS,vS_prime, dAlpha,vParC):
      if mS.ndim == 1:
        mS = mS.reshape(1, 2)
        
      vK = mS[:,0]
      vM = mS[:,1]
      dM_prime = vS_prime[1]
      da1,da2,db1,db2,dGamma,dPhi = vParC 
      
      vMu = 1-(vK**(-dAlpha)/dGamma)*(dM_prime-(1-dPhi)*vM)
      vMu = np.where((vMu >= 0) & (vMu <= 1), vMu, -np.inf)
      vDnm = 1+db1*vM**db2
      
      vY = np.divide((vK**dAlpha*(1-da1*vMu**da2)),vDnm)
      
      return [vY, vMu]

def fUtilde(mS,vS_prime, dDelta,dAlpha,dSigma,vParC):
    iN = mS.shape[0]
    vProd,vMu = fProd(mS,vS_prime,dAlpha,vParC)
    
    vK = mS[:,0]
    vM = mS[:,1]
   
    vU_tilde = -np.ones(iN)*np.inf
    bMu = 1-(vMu < 0)-(vMu > 1)
    
    idx = np.where(bMu == 1)[0]
    
    vK = vK[idx]
    vM = vM[idx]
    vProd = vProd[idx]
    
    dK_prime, dM_prime = vS_prime
    vK_d = (1-dDelta)*vK
    vU_arg = vProd + vK_d-dK_prime
    bU = 1-(vU_arg < 0)
    idx2 = np.where(bU == 1)[0]
    
    vU_tilde[idx[idx2]] =fUtility(dSigma,vU_arg[idx2])
    #print(vU_tilde)
    return vU_tilde

def calc_utility(S, phi, alpha, gamma, delta, a1, a2, b1, b2):
    K = S[:,0]
    M = S[:,1]

    ratio1 = (1-phi)*M/K**alpha    
    ratio2 = M/(K**alpha)[:, np.newaxis]
    result = ratio2 - ratio1[:, np.newaxis]
    mu = 1-(1/gamma)*result
    mu_mat = np.where((mu >= 0) & (mu <= 1), mu, -np.inf)
    
    c_1 = 1-a1*(mu_mat**a2)
    c_2 = K**alpha /(1+b1* M **b2)
    c_3 = c_2[:, np.newaxis] * c_1
    vdK = (1-delta)*K
    c_4 = c_3 + vdK[:, np.newaxis]
    c = c_4 - K
    print('Yess')
    #print(c)
    print('Nooo')
    c_final = np.where((c > 0), c, -np.inf)
    #condition = (c < 0) & (c != -np.inf)
    #count = np.sum(condition)
    U_mat = c_final.copy()
    U_mat[U_mat >= 0] = np.log(U_mat[U_mat >= 0])
    #print(U_mat)
    return U_mat

###########################################################

def main():
    alpha = 1/3
    beta = 0.75  
    delta = 0.5  
    sigma = 1  
    a1 = 0.06
    a2 = 2.8
    b1 = 0.5
    b2 = 2
    gamma = 0.13
    phi = 0.06
    nK = 100
    nM = 100
    K1, KnK = 0.1, 0.4 
    M1, MnM = 0, 0.5  
    epsilon = 1e-4 

    K_grid = np.linspace(K1, KnK, nK)
    M_grid = np.linspace(M1, MnM, nM)
    K_mesh, M_mesh = np.meshgrid(K_grid, M_grid)
    S = np.c_[K_mesh.ravel(), M_mesh.ravel()]



    vK_range = np.array([0.1,0.4])
    vM_range = np.array([0,0.5])
    iN_K = nK
    iN_M = nM
        
    vGridK = GetGrid(vK_range, iN_K)
    vGridM = GetGrid(vM_range, iN_M)   
    mMesh = np.meshgrid(vGridK,vGridM)
    mS = np.array(mMesh).T.reshape(-1,2)
    #print(mS)

    #U_mat1 = calc_utility(S, phi, alpha, gamma, delta, a1, a2, b1, b2)
    #print(U_mat1)
    U_mat2 = calc_utility(mS, phi, alpha, gamma, delta, a1, a2, b1, b2)
    #print(U_mat2[0,:])
    U_mat = U_mat2

    n = U_mat.shape[0]
    V_0 = np.zeros(n)
    pi_0 = np.zeros(n)
    iteration = 0
    beta_75 = 0.75 
    beta_85 = 0.85 
    V, pi_75 = iterate(V_0, pi_0, U_mat, beta_75, epsilon, iteration, n)
    #np.savetxt("pi_75.csv", pi_75, delimiter=",")
    #V, pi_85 = iterate(V_0, pi_0, U_mat, beta_85, epsilon, iteration, n)
    #np.savetxt("pi_85.csv", pi_85, delimiter=",")
    #print(pi_75)
    T = 100
    K_path_75, M_path_75, mu_path_75, c_path_75 = calc_paths(mS, K_grid, M_grid, pi_75, T, a1, a2, b1, b2, phi, alpha, delta, gamma)
    print(K_path_75)


    dSigma = 1
    dBeta = 0.75
    dDelta = 0.5
    dAlpha = 1/3
    
    #Climate-specific parameters
    db1 = 0.5
    db2 = 2
    da1 = 0.06
    da2 = 2.8
    dGamma = 0.13
    dPhi = 0.06
    
    vParC = np.array([da1,da2,db1,db2,dGamma,dPhi])
    iT = 100    
    dK_0 = 0.1
    dM_0 = 0
    
    vInit = np.array([dK_0,dM_0])
    
    vK_range = np.array([0.1,0.4])
    vM_range = np.array([0,0.5])
    
    iN = iN_K*iN_M
    
    vGridK = GetGrid(vK_range, iN_K)
    vGridM = GetGrid(vM_range, iN_M)
    
    #Initial value for value function V
    vV_0 = np.zeros(iN)
    #Tolerance for the value function algorithm
    dEps = 10**-4
    
    #[mS,mUmat] = ComputeUmat(vGridK, vGridM, dDelta, dAlpha, dBeta, dSigma,vParC)
    #print(mUmat[0,:])
    V, pi_75 = iterate2(V_0, pi_0, mUmat, beta_75, epsilon, iteration, n)
    #print(pi_75)


###########################################################
### call main
if __name__ == "__main__":
    main()