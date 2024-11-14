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
def plot_excess(M_path_75, M_path_85):
    plt.figure(figsize=(10, 6))
    plt.plot(M_path_75, label='M(t)|'+r' $\beta=0.75$')
    plt.plot(M_path_85, label='M(t)|'+r' $\beta=0.85$')
    plt.xlabel("Time Period")
    plt.ylabel("Value")
    plt.title("Paths of Excess Carbon M")
    plt.legend()
    plt.grid()
    plt.savefig('Excess.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_abatement(mu_path_75,mu_path_85):
    plt.figure(figsize=(10, 6))
    plt.plot(mu_path_75, label=r' $\mu(t)| \beta=0.75$')
    plt.plot(mu_path_85, label=r' $\mu(t)| \beta=0.85$')
    plt.xlabel("Time Period")
    plt.ylabel("Value")
    plt.title("Paths of Abatement mu")
    plt.legend()
    plt.grid()
    plt.savefig('Abatement.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_capital(K_path_75, K_path_85):
    plt.figure(figsize=(10, 6))
    plt.plot(K_path_75, label='K(t)|'+r' $\beta=0.75$')
    plt.plot(K_path_85, label='K(t)|'+r' $\beta=0.85$')
    plt.xlabel("Time Period")
    plt.ylabel("Value")
    plt.title("Paths of Capital")
    plt.legend()
    plt.grid()
    plt.savefig('Capital.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_consumption(c_path_75, c_path_85):
    plt.figure(figsize=(10, 6))
    plt.plot(c_path_75, label='c(t)|'+r' $\beta=0.75$')
    plt.plot(c_path_85, label='c(t)|'+r' $\beta=0.85$')
    plt.xlabel("Time Period")
    plt.ylabel("Value")
    plt.title("Paths of Consumption")
    plt.legend()
    plt.grid()
    plt.savefig('Consumption.png', dpi=300, bbox_inches='tight')
    plt.show()

def calc_paths(S, K_grid, M_grid, pi, T, a1, a2, b1, b2, phi, alpha, delta, gamma):
    '''
    All paths are computed following given theory
    '''
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

###########################################################

def main():
    alpha = 1/3 
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
    K_grid = np.linspace(K1, KnK, nK)
    M_grid = np.linspace(M1, MnM, nM)
    K_mesh, M_mesh = np.meshgrid(K_grid, M_grid)
    S = np.c_[K_mesh.ravel(), M_mesh.ravel()]
    pi_75 = pd.read_csv("pi_75.csv", header=None).values.astype(int)[:,0]
    pi_85 = pd.read_csv("pi_85.csv", header=None).values.astype(int)[:,0]
    print(pi_75)

    T=100
    K_path_75, M_path_75, mu_path_75, c_path_75 = calc_paths(S, K_grid, M_grid, pi_75, T, a1, a2, b1, b2, phi, alpha, delta, gamma)
    K_path_85, M_path_85, mu_path_85, c_path_85 = calc_paths(S, K_grid, M_grid, pi_85, T, a1, a2, b1, b2, phi, alpha, delta, gamma)


    plot_capital(K_path_75, K_path_85)
    plot_excess(M_path_75,M_path_85)
    plot_consumption(c_path_75,c_path_85)
    plot_abatement(mu_path_75,mu_path_85)


###########################################################
### call main
if __name__ == "__main__":
    main()