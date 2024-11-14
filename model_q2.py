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
def iterate(V_0, pi_0, u_mat, beta, epsilon, i, n):
    '''
    Value iteration, but this time row wise. Since a 10 000 by 10 000 matrix takes a lot of storage data
    '''
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

def calc_utility(S, phi, alpha, gamma, delta, a1, a2, b1, b2):
    """
    derive utility in a matrix multiplication method ensuring fast computing time
    """
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
    c_final = np.where((c > 0), c, -np.inf)
    U_mat = c_final.copy()
    U_mat[U_mat >= 0] = np.log(U_mat[U_mat >= 0])
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

    U_mat = calc_utility(S, phi, alpha, gamma, delta, a1, a2, b1, b2)

    n = U_mat.shape[0]
    V_0 = np.zeros(n)
    pi_0 = np.zeros(n)
    iteration = 0
    beta_75 = 0.75 
    beta_85 = 0.85 
    V, pi_75 = iterate(V_0, pi_0, U_mat, beta_75, epsilon, iteration, n)
    np.savetxt("pi_75.csv", pi_75, delimiter=",")
    V, pi_85 = iterate(V_0, pi_0, U_mat, beta_85, epsilon, iteration, n)
    np.savetxt("pi_85.csv", pi_85, delimiter=",")

###########################################################
### call main
if __name__ == "__main__":
    main()