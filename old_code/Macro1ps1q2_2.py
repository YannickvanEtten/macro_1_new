# -*- coding: utf-8 -*-
"""
Macro1ps1q2.py

Purpose:
    Script to compute value function, policy function, and implied optimal
    consumption and price paths
    
Student numbers:
    AAA, BBB, CCC, DDD
    
Date:
    10 Nov 2024
"""

import numpy as np 
import time
import matplotlib.pyplot as plt 

def fUtility(dSigma, dC):
    """
    Purpose: Compute utility value given sigma and c
    
    Inputs:
    dSigma: double input for sigma
    dC: double input for c 
    
    Return value:
    dU: double utility value
    """

    if dSigma == 1:
      dU = np.log(dC)
    else:
      dU = (dC**(1-dSigma)-1)/(1-dSigma)
    
    return dU 

def mUtility(dSigma,dC):
    """
    Purpose: compute marginal utility value given sigma and c 
    
    Inputs:
        dSigma double for sigma
        dC double for c
        
    Return value:
        dMU: double for marginal utility value
    """
    dMU = dC**-dSigma
    
    return dMU

def fProd(mS,vS_prime, dAlpha,vParC):
      """
      Purpose: Compute production function value given alpha and K (Cobb-Douglas)
    
      Inputs:
      mS: iN x 2 matrix containing all K (capital) and M (carbon)
      vS' vector of length 2 containing a single K' (capital) and M' (carbon)
      dAlpha: double input for alpha
      vParC: vector of climate-specific parameters (6 x 1)
    
      Return value:
      dY: double output value
      """
      if mS.ndim == 1:
        mS = mS.reshape(1, 2)
        
      vK = mS[:,0]
      vM = mS[:,1]
      dM_prime = vS_prime[1]
      da1,da2,db1,db2,dGamma,dPhi = vParC 
      
      vMu = 1-(vK**(-dAlpha)/dGamma)*(dM_prime-(1-dPhi)*vM)
      vDnm = 1+db1*vM**db2
      
      vY = np.divide((vK**dAlpha*(1-da1*vMu**da2)),vDnm)
      
      return [vY, vMu]
  
def GetGrid(vRange,iN):
    """
    Purpose: construct a grid of values (equally spaced)
    
    Inputs:
        vRange: 2x1 vector containing the min and max of K for our grid
        iN: integer for the grid length
        
    Return value:
        vGrid: iN vector representing the grid
    
    """
    [dK_min, dK_max] = vRange
    vGrid = np.linspace(dK_min,dK_max,iN)
    
    return vGrid

def fUtilde(mS,vS_prime, dDelta,dAlpha,dSigma,vParC):
    """
    Purpose: compute u_tilde(i,j), the i,j-th element of Umat
    
    Inputs:
        mS: iN x 2 matrix containing all (K,M) 
        vS_prime vector of length 2 containing a single (K',M')
        dDelta,dAlpha,dSigma: user-specified parameters (double)
        vParC: vector of climate specific parameters (user-specified)
        
    Return value:
        vU_tilde: vector: column j of u_tilde
    """
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
        
    return vU_tilde

def ComputeUmat(vGrid1,vGrid2,dDelta,dAlpha,dBeta,dSigma, vParC):
    """
    Purpose: compute the iN x iN matrix Umat for two state variables
    
    Inputs:
        dDelta,dAlpha,dBeta,dSigma: double user-specified parameters
        vParC: vector of climate specific parameters
        vGrid1: grid of possible values for first state variable (iN_1 vector)
        vGrid2: grid of possible values for first state variable (iN_2 vector)
            
    Return value:
        mUmat: iN x iN matrix of u_tilde(K,K',M,M') combinations
    """
    iN_1 = len(vGrid1)
    iN_2 = len(vGrid2)
    
    iN = iN_1 * iN_2
    mMesh = np.meshgrid(vGrid1,vGrid2)
    #Gives all combinations of vectors of length 2 from our 2 grids 
    mS = np.array(mMesh).T.reshape(-1,2)
    mUmat = np.zeros((iN,iN))

    for j in range(iN):
        vS_prime = mS[j,:]
        mUmat[:,j] = fUtilde(mS,vS_prime,dDelta,dAlpha,dSigma,vParC)
    return [mS,mUmat]

def GetVandPi(vV_0,dEps,dBeta,mUmat):
    """
    Purpose: compute the value function and policy function
    
    Inputs:
        vV_0 initial guess for the value function (iN vector)
        dEps tolerance value (double)
        dBeta discount factor (double)
        mUmat iN x iN matrix Umat (see notes for definition)
        
    Return value:
        vV value function 
        vPi policy function
        Stored in list called 'out'
    """
    iN = len(vV_0)
    mV_0 = np.tile(vV_0,(iN,1))
    
    vV = np.max(mUmat+dBeta*mV_0,axis = 1)
    #vPi_1 = np.argmax(mUmat+dBeta*mV_0,axis = 0)
    
    while np.linalg.norm(vV_0-vV, ord = 2) > dEps:
        print('x') # sign of life 
        vV_0 = vV
        mV_0 = np.tile(vV_0,(iN,1))
        vV = np.max(mUmat+dBeta*mV_0,axis = 1)
   
    vPi = np.argmax(mUmat+dBeta*mV_0, axis = 1)
    
    out = [vV,vPi]
    
    return out 
  
def PlotOptimalPolicy(vPi,mS,iT,vInit):
    """
    Purpose: plot the optimal capital path using the policy function
    
    Inputs:
        vPi policy function evaluated at our iN grid points (iN vector)
        mS iN x 2 matrix of K,M combinations 
        iT  number of time periods (integer)
        vInit Initial capital and carbon (2 x 1 vector)
        
    Return value:
        mS_opt iT x 2 matrix for optimal capital and carbon 
    """
    mS_opt = np.zeros((iT,2))
    
    idx = np.argmin(np.linalg.norm(mS-vInit, ord = 2, axis = 1))
    mS_opt[0,:] = mS[idx,:]
    
    for t in range(1,iT):
        idx = vPi[idx]
        mS_opt[t,:] = mS[idx,:]
    print(mS_opt[:,0])
    plt.figure(figsize = (12,6))
    plt.plot(1+np.arange(iT), mS_opt[:,0], 'r+')
    plt.plot(1+np.arange(iT), mS_opt[:,1], 'k+')
    plt.legend(['$K(t)$','$M(t)$'])
    plt.xlabel('$t$')
    plt.ylabel('$K(t)$,$M(t)$')
    plt.title('Optimal paths for $K(t)$ and $M(t)$')
             
    return mS_opt 

def PlotOptimalC(mS_opt,dDelta,dAlpha, vParC):
    """
    Purpose: compute optimal consumption path and plot the result
    
    Inputs:
        mS_opt: iTx2 matrix for optimal K,M paths 
        dDelta: depreciation rate (double)
        dAlpha: Cobb-Douglas parameter (double)
        vParC: vector of climate-specific parameters (6x1) 
        
    Return value:
        vC_opt: iT vector representing optimal consumption path
    """
    iT = mS_opt.shape[0] 
    vC_opt = np.zeros(iT-1)
    vMu = np.zeros(iT-1)
    
    for t in range(iT-1):
        vS = mS_opt[t,:]
        vS_prime = mS_opt[t+1,:]
        dProd,dMu= fProd(vS, vS_prime, dAlpha, vParC)
        vC_opt[t] = dProd+(1-dDelta)*vS[0]-vS_prime[0]
        vMu[t] = dMu
    #print(dProd)
    #print(vC_opt)
        
    plt.figure(figsize = (12,6))
    plt.plot(1+np.arange(iT-1),vC_opt,'b+')
    plt.plot(1+np.arange(iT-1),vMu, 'g+')
    plt.xlabel('$t$')
    plt.ylabel('$c(t)$, $\mu(t)$')
    plt.legend(['$c(t)$', '$\mu(t)$'])
    plt.title('Optimal paths for $c(t)$ and $\mu(t)$')
    
    return vC_opt

def PlotOptimalP(vC_opt,dBeta,dSigma):
    """
    Purpose: compute and plot the optimal price path
    
    Inputs:
        vC_opt: optimal consumption path (iT-1 vector)
        dBeta: discount factor (double)
        dSigma: utility-specific parameter (double)
        
    Return value:
        vP_opt iT-2 vector representing the optimal price path
    """
    iT = len(vC_opt)+1 #Consumption path 1 shorter than capital path
    vP_opt = np.zeros(iT-2)
    
    for t in range(iT-2):
        dMU0 = mUtility(dSigma,vC_opt[t])
        dMU1 = mUtility(dSigma,vC_opt[t+1])
        vP_opt[t] = dBeta*(dMU1/dMU0)
    
    plt.figure(figsize = (12,6))
    plt.plot(1+np.arange(iT-2),vP_opt,'k+')
    plt.xlabel('$t$')
    plt.ylabel('$p(t+1|t)$')
    plt.title('Optimal path for $p(t+1|t)$')
    
    return vP_opt
    
def main():
    #Sigma >0, != 1
    start = time.time()
    
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
    
    #Starting capital:
    dK_0 = 0.1
    #Initial carbon:
    dM_0 = 0
    
    vInit = np.array([dK_0,dM_0])
    
    vK_range = np.array([0.1,0.4])
    vM_range = np.array([0,0.5])
    iN_K = 100
    iN_M = 100
    
    iN = iN_K*iN_M
    
    vGridK = GetGrid(vK_range, iN_K)
    vGridM = GetGrid(vM_range, iN_M)
    
    #Initial value for value function V
    vV_0 = np.zeros(iN)
    #Tolerance for the value function algorithm
    dEps = 10**-4
    
    [mS,mUmat] = ComputeUmat(vGridK, vGridM, dDelta, dAlpha, dBeta, dSigma,vParC)
    print(mUmat[0,:])
    
    out = GetVandPi(vV_0, dEps, dBeta, mUmat)
    vV = out[0]
    vPi = out[1]
    
    mS_opt = PlotOptimalPolicy(vPi,mS,iT,vInit)
    vC_opt = PlotOptimalC(mS_opt,dDelta,dAlpha,vParC)
    #vP_opt = PlotOptimalP(vC_opt, dBeta, dSigma)
    
    finish = time.time()
    print('Run time: ', finish-start, 'seconds')
if __name__ == '__main__':
    main()
    