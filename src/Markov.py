
import numpy as np
import pandas as pd
import os

dirname = os.path.dirname(__file__) #relative path


def foward(s, b): #foward algorithm 
    alpha = np.zeros((2,len(s))) #init vects
    eta = np.zeros(len(s))
    
    alpha[:,0] = pi*b[int(s[0]*10),:]
    eta[0] = np.sum(alpha[:,0])
    alpha[:,0] = alpha[:,0]/eta[0]

    for T in range(0, len(s)-1):
        for j in range(0,2):
            for i in range(0,2):
                alpha[j,T+1] = alpha[j,T+1] + alpha[i,T]*a[i,j]*b[int(s[T+1]*10),j]#
        eta[T+1] = np.sum(alpha[:,T+1])
        alpha[:,T+1] = alpha[:,T+1]/eta[T+1]
    return alpha, eta

def backward(s, b, eta):
    beta = np.zeros((2,len(s)))
    beta[1,-1] = 1
    for T in range(len(s)-2, -1, -1): 
        for i in range(0,2):
            for j in range(0,2):
                beta[i,T] = beta[i,T] + beta[j,T+1]*a[i,j]*b[int(s[T+1]*10),j]#
        beta[:,T] = beta[:,T]/eta[T+1]  
    return beta
def get_gamma(alpha, beta, s):
    gamma = np.zeros((2,len(s)))
    for t in range(0,len(s)):
        den  = 0
        den = np.sum(alpha[:,t]*beta[:,t])
        for i in range(0,2):
            gamma[i, t] = alpha[i, t]*beta[i, t]/den
    
    return gamma
def get_zeta(alpha, beta, s, a, b, eta):
    zeta = np.zeros((2,2, len(s)-1 ))
    for t in range(0, len(s)-1):
        for i in range(0,2):
            for j in range(0,2):
                zeta[i, j, t] = alpha[i,t]*a[i,j]*b[int(s[t+1]*10),j]*beta[j, t+1]/(eta[t+1]*np.sum(alpha[:,t]*beta[:,t]))
    return zeta
def setNewTransProb(zeta):
    temp1 = np.sum(zeta, axis = 2) #temporary variable
    temp2 = np.sum(zeta, axis = 1)
    a_new = np.zeros((2,2)) #new array the new prob transition
    
    for i in range(0,2):
        for j in range(0,2):
            a_new[i, j] = temp1[i, j]/np.sum(temp1, axis = 1)[i]
    a = np.copy(a_new)
    return a
    
def setNewVisableProbability(gamma, s):
    u1 = np.sum(gamma[0,:]*s[:,0])/np.sum(gamma[0,:])
    u2 = np.sum(gamma[1,:]*s[:,0])/np.sum(gamma[1,:])
    c1 = np.sum(gamma[0,:]*abs(s[:,0]-u1))/np.sum(gamma[0,:])
    c2 = np.sum(gamma[1,:]*abs(s[:,0]-u2))/np.sum(gamma[1,:])
    x = np.arange(0,21.1,0.1)
    
    g1 = 1/(c1*np.sqrt(2*np.pi))*np.exp(-1/2*pow((x-u1)/c1, 2))
    g2 = 1/(c2*np.sqrt(2*np.pi))*np.exp(-1/2*pow((x-u2)/c2, 2))
    
    b = np.array([g1/np.sum(g1),g2/np.sum(g2)]).T
    return b
filepath = dirname+'/../content';
a = pd.read_csv(filepath+'/distribution/a.csv')#probabilidade de mudar de estado
b = pd.read_csv(filepath+'/distribution/b.csv')#probabilidade dos eventos de velocidade
M = pd.read_csv(filepath+'/embarcacoes/embacacoes.csv') #all files
M = M.values
a = a.values
b = b.values
pi = np.array([0,1])#variavel da probabilidade
s = M[0:,7] #vetor de velocidade do trajeto N
s = np.array([s]).T

for i in range(0,10):
    alpha, eta = foward(s, b)
    beta = backward(s, b, eta)
    gamma = get_gamma(alpha, beta, s)
    zeta = get_zeta(alpha, beta, s, a, b, eta)
    a = setNewTransProb(zeta)
    b = setNewVisableProbability(gamma, s)

print(a)