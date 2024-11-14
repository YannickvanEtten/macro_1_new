'''
Filename: model_qi.py

Purpose:
    
Date:
    13 November 2024
Author:
     
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

def iterate(V_0, pi_0, u_mat, beta, epsilon, n):
    V_mat = np.tile(V_0, (n, 1))
    V_1 = np.max(u_mat + beta * V_mat, axis=1) 
    pi = np.argmax(u_mat + beta * V_mat, axis=1)
    if np.linalg.norm(V_1 - V_0) < epsilon:
        return V_1, pi
    return iterate(V_1, pi_0, u_mat, beta, epsilon, n)

def F(K, alpha):
    return K**alpha 

def plot_capital(V, pi, K_grid, K0, T):
    current_index = 0
    K_path = [K_grid[current_index]]
    for t in range(T):
        next_index = pi[current_index] 
        K_path.append(K_grid[next_index])
        current_index = next_index  

    plt.plot(K_path)
    plt.xlabel("Time Period")
    plt.ylabel("Capital Stock")
    plt.title("Optimal Path of Capital")
    plt.show()
    return K_path

def plot_consumption(K_path, alpha, delta):
    C_path = []
    for t in range(len(K_path) - 1):
        K_t = K_path[t]
        K_next = K_path[t + 1]
        C_t = K_t**alpha + (1 - delta) * K_t - K_next  # F(K_t, 1) = K_t^alpha
        C_path.append(C_t)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(C_path)), C_path, marker='o', markersize=3)
    plt.xlabel("Time Period")
    plt.ylabel("Consumption")
    plt.title("Path of Consumption")
    plt.grid(True)
    plt.show()
    return C_path

def plot_prices(C_path, beta):
    P_path = []  
    for t in range(len(C_path) - 1):
        C_t = C_path[t]
        C_next = C_path[t + 1]
        P_t = beta * (C_t / C_next) 
        P_path.append(P_t)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(P_path)), P_path, marker='o', markersize=3)
    plt.xlabel("Time Period")
    plt.ylabel("Price")
    plt.title("Path of Price (p[t+1 | t])")
    plt.grid(True)
    plt.show()

###########################################################

def main():
    beta =  0.984
    sigma = 1
    alpha = 1/3
    delta = 0.025
    epsilon = 0.0001
    n = 1000

    K_star = steady_state_cap(alpha,beta,delta)
    V_0 = np.zeros(n)
    pi_0 = np.zeros(n)
    K0 = 0.75 * K_star
    K1 = 0.5 * K_star
    Kn = 01.5 * K_star
    K_grid = np.linspace(K1, Kn, n)
    U_mat = np.full((n, n), -np.inf)
    for i in range(n):
        for j in range(n):
            lower_bound = (1 - delta) * K_grid[i]
            upper_bound = F(K_grid[i],alpha) + lower_bound
            if lower_bound <= K_grid[j] <= upper_bound: # Check if K_j' is within G(K_i)
                c = F(K_grid[i],alpha) + (1 - delta) * K_grid[i] - K_grid[j]                
                if c > 0:
                    U_mat[i, j] = np.log(c)

    V, pi = iterate(V_0, pi_0, U_mat, beta, epsilon, n)
    T = 250
    K_path = plot_capital(V, pi, K_grid, K0, T)
    C_path = plot_consumption(K_path, alpha, delta)
    plot_prices(C_path, beta)

###########################################################
### call main
if __name__ == "__main__":
    main()