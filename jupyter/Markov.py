import numpy as np
import pandas as pd

class HMM:
    
    @staticmethod
    def backward(s, b, a, eta):
        beta = np.zeros((2,len(s)))
        beta[1,-1] = 1
        
        for T in range(len(s)-2, -1, -1): 
            for i in range(0,2):
                for j in range(0,2):
                    beta[i,T] = beta[i,T] + beta[j,T+1]*a[i,j]*b[int(s[T+1]*10),j]#
            beta[:,T] = beta[:,T]/eta[T+1]  
        return beta
    
    @staticmethod
    def forward(s, a, b, pi):
        alpha = np.zeros((2,len(s)))
        eta = np.zeros(len(s))
    
        alpha[:,0] = pi*b[int(s[0]*10),:]
        eta[0] = np.sum(alpha[:,0])
        alpha[:,0] = alpha[:,0]/eta[0]
        #print(alpha[:,0])
        for T in range(0, len(s)-1):
            for j in range(0,2):
                for i in range(0,2):
                    alpha[j,T+1] = alpha[j,T+1] + alpha[i,T]*a[i,j]*b[int(s[T+1]*10),j]#
            eta[T+1] = np.sum(alpha[:,T+1])
            alpha[:,T+1] = alpha[:,T+1]/eta[T+1]
    
        return alpha, eta
   
    @staticmethod
    def get_gamma(s, alpha, beta):
        gamma = np.zeros((2,len(s)))
        for t in range(0,len(s)):
            den  = 0
            den = np.sum(alpha[:,t]*beta[:,t])
            for i in range(0,2):
                gamma[i, t] = alpha[i, t]*beta[i, t]/den            
        return gamma
    
    @staticmethod
    def get_zeta(s, alpha, beta, eta, a, b):
        zeta = np.zeros((2,2, len(s)-1 ))
        for t in range(0, len(s)-1):
            for i in range(0,2):
                for j in range(0,2):
                    zeta[i, j, t] = alpha[i,t]*a[i,j]*b[int(s[t+1]*10),j]*beta[j, t+1]/(eta[t+1]*np.sum(alpha[:,t]*beta[:,t]))
        return zeta
    
    @staticmethod
    def get_distribution()