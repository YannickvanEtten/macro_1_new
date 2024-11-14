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
def steady_state_cap(alpha,beta,delta):
    return(((1 + beta * (delta - 1)) / (beta * alpha))**(1 / (alpha - 1)))

def get_utility(consumption, sigma):
    if (sigma == 1):
        return np.log(consumption)
    return (consumption ** (1 - sigma) - 1) / (1 - sigma)

def iterate(V_0, pi_0, u_mat, beta, epsilon, i, n):
    V_mat = np.tile(V_0, (n, 1))
    V_1 = np.max(u_mat + beta * V_mat, axis=1) 
    pi = np.argmax(u_mat + beta * V_mat, axis=1)
    if np.linalg.norm(V_1 - V_0) < epsilon:
        return V_1, pi
    print(i, np.linalg.norm(V_1 - V_0))
    i += 1
    return iterate(V_1, pi_0, u_mat, beta, epsilon, i, n)

def F(K, alpha):
    return K**alpha 

def utility(c, sigma):
    if sigma == 1:
        return np.log(c)
    else:
        return (c ** (1 - sigma) - 1) / (1 - sigma)

def consumption(K_t, M_t, K_next, M_next, b1, b2, a1, a2, gamma, alpha, delta, phi):
    D = 1 + b1 * M_t**b2
    abatement_term = 1 - a1 * ((M_next - (1 - phi) * M_t) / (gamma * K_t**alpha))**a2
    net_production = K_t**alpha / D
    return net_production * abatement_term + (1 - delta) * K_t - K_next

def compute_utility_matrix(K_grid, M_grid, phi, gamma, alpha, S, b1, b2, a1, a2, delta, sigma):
    n = len(K_grid) * len(M_grid)
    U_mat = np.full((n, n), -np.inf)  
    for i in range(n):
        K_t, M_t = S[i]
        print(i)
        for j in range(n):
            K_next, M_next = S[j]
            c_t = consumption(K_t, M_t, K_next, M_next, b1, b2, a1, a2, gamma, alpha, delta, phi)
            if c_t > 0 and 0 <= (M_next - (1 - phi) * M_t) / (gamma * K_t**alpha) <= 1:
                U_mat[i, j] = utility(c_t, sigma)
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
    print(S)

    K = S[:,0]
    M = S[:,1]
    #print((M - (1 - phi) * M.T))
    #test = (M - (1 - phi) * M.T) / (gamma * K**alpha)
    ratio1 = (1-phi)*M/K**alpha
    ratio2 = M.reshape(-1, 1)/(K**alpha)
    print(ratio2)

    #U_mat = compute_utility_matrix(K_grid, M_grid, phi, gamma, alpha, S, b1, b2, a1, a2, delta, sigma)
    #print(U_mat)




###########################################################
### call main
if __name__ == "__main__":
    main()