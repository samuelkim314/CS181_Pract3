import csv
import sys
import itertools
import numpy as np

def importFruitData(filename="fruit.csv"):
    reader = csv.reader(open(filename, 'rb'))
    fruit = []
    xs = []
    for row in itertools.islice(reader,1,None):
        fruit.append(float(row[0]))
        x1 = float(row[1])
        x2 = float(row[2])
        xs.append([x1, x2])

    return np.array(xs), np.array(fruit)

"""
N: # of inputs
M: # of features
K: # of classes
w (KxM): w_km is weights for kth class, mth feature
y (NxK): y_nk is posterior probability for nth input, kth class
phi (NxM): phi_nm is basis function for nth input, mth feature
t (N): output for nth input
"""
def logRegress(xs, y):
    pass

"""returns phi
no function is applied, but 1 is appended as a feature for the bias
"""
def basisNone(x):
    phi = np.insert(x, 0, 1, axis=1)
    return phi

""" Calculates posterior probabilities,
y_k(phi)=exp(a_k)/Sum_j(exp(a_j))
where a_k = w_k * phi
TODO: Vectorize
"""
def y(w, phi):
    # array of exp(a_k)
    expA = np.empty(len(w))
    #array of y_k(phi)
    y = np.empty(len(w))
    #Calculates exp(a_k)
    for k in range(len(w)):
        expA[k] = math.exp(np.dot(w[k],phi))
    sum = np.sum(expA)
    for k in range(len(w)):
        y[k] = math.exp(expA[k]) / sum
    return y

"""Calculates the Hessian matrix
"""
def hessMat(y, phi):
    m = y.shape[1]
    hess = np.empty((m,m))
    i = np.identity(m)
    for k in range(m):
        for j in range(m):
            for n in range(len(phi)):
                temp[n] = y[n,k]*(i[k,j]-y[n,j])*np.outer(phi[n],phi[n])
                hess[j,k] = np.sum(temp)

    return hess

"""Gradient of the error function with respect to each of w_j
"""
def gradE(y, t, phi):
    grad = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        for n in range(len(phi)):
            temp[n] = (y[n,j] - t[n,j])*phi[n]
            grad[j] = sum(temp[n])
    return grad

"""Updates w based on the Newton-Raphson iterative optimization (IRLS)
"""
def newW(w, y, t, phi):
    return w - np.dot(np.linalg.inv(hessMat(y, phi)), gradE(y, t, phi))