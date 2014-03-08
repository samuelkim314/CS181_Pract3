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

def logRegress(xs, y):
    pass

def basisNone(x):
    ones = np.ones(x.shape[1])
    phi = np.insert(x, 0, ones)
    return phi

def grad(ys, ts, phi):
    pass

""" Calculates posterior probabilities,
y_k(phi)=exp(a_k)/Sum_j(exp(a_j))
where a_k = w_k * phi
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

def gradE(y, t, phi):
    grad = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        for n in range(len(phi)):
            temp[n] = (y[n,j] - t[n,j])*phi[n]
            grad[j] = sum(temp[n])
    return grad

def newW(w, y, t, phi):
    return w - np.dot(np.linalg.inv(hessMat(y, phi)), gradE(y, t, phi))