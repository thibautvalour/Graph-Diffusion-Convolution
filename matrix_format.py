import numpy as np
import math as math


def gdc_pagerank(A, alpha, eps):
    N = A.shape[0]

    # Self-loops 
    A_loop = A + np.eye(N)
    
    # Symmetric transition matrix
    D_loop = np.sum(A_loop, axis=1)
    D_sqrt_inv = np.diag(1/np.sqrt(D_loop))

    T_sym = D_sqrt_inv @ A_loop @ D_sqrt_inv

    # PPR-based diffusion
    S = alpha * np.linalg.inv(np.eye(N) -(1-alpha)*T_sym)

    # Sparcify
    S[S < eps] = 0

    D_tilde = np.sum(S, axis=1)
    D_tilde_inv = np.diag(1/D_tilde)
    T_S = S @ D_tilde_inv
    
    return T_S


def gdc_heat(A, t, sum_limit, eps):

    N = A.shape[0]  
    
    A_loop = A + np.eye(N)
    D_loop = np.sum(A_loop, axis=1)
    D_sqrt_inv = np.diag(1/np.sqrt(D_loop))

    T_sym = D_sqrt_inv @ A_loop @ D_sqrt_inv

    S = np.zeros((N, N))
    temp_mat = np.eye(N)
    for k in range(sum_limit):
        heat_coeff = math.exp(-t) * t**k / math.factorial(k)
        S += heat_coeff * temp_mat
        temp_mat = temp_mat @ T_sym

    # Sparcification
    S[S < eps] = 0

    D_tilde = np.sum(S, axis=1)
    D_tilde_inv = np.diag(1/D_tilde)
    T_S = S @ D_tilde_inv
    
    return T_S

def compute_Lsym(A):
    N = A.shape[0]
    D = np.sum(A, axis=1)
    D_sqrt_inv = np.diag(1/np.sqrt(D))
    Tsym = D_sqrt_inv @ A @ D_sqrt_inv

    Lsym = np.eye(N) - Tsym
    return Lsym

def compute_Lrw(A):
    N = A.shape[0]
    D = np.sum(A, axis=1)
    D_inv = np.diag(1/D)
    Lrw = np.eye(N) - D_inv @ A
    return Lrw
