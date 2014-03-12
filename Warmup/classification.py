import math
import numpy as np
import numpy.ma as ma
import scipy.linalg



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Generative classifier with Bayesian class-conditional densities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def class_split(x):
    """
    return a list of numpy arrays with datapoints belonging only to that class
    """
    return np.split(x, [19,43])

def mean(x, t, K):
    """
    x:      vector with dimension N x (M - 1)
    t:      vector with dimension N

    N:      vector with dimension K
    mu:     vector with dimension K x (M - 1)
    """
    N = np.zeros(K)
    mu = np.zeros((K, x.shape[1]))
    for i in range(x.shape[0]):
        k = t[i] - 1
        mu[k] = np.add(mu[k], x[i])
        N[k] += 1
    return N, np.divide(mu, N[:,None])

def covariance(x, mu):
    """
    x:      vector with dimension N x (M - 1)
    mu:     vector with dimension K x (M - 1)

    S:      matrix
    """




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Multiclass logistic regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def logisticRegression(x, t, K, alpha=0.01, epsilon=0.001, steps=500):
    """
    Return  w:      vector with dimension M(K - 1)
    """
    #initialize w, phi, _t
    phi = basisNone(x)
    for _ in range(K - 2):
        phi = scipy.linalg.block_diag(phi, basisNone(x))
    w = np.zeros(phi.shape[1])

    _t = np.empty(0)
    for k in range(K - 1):
        _t = np.append(_t, ma.masked_not_equal(ma.masked_not_equal(t, k + 1).filled(0), 0).filled(1))

    error = 0
    de = 0
    #gradient descent on error function
    for step in range(steps):
        previous = error
        _y = y(w, phi)
        w = w_new(w, _y, _t, phi, K, alpha)
        error = np.dot(-_t, np.array(map(math.log, _y)))
        de = error - previous
        if abs(de) < epsilon:
            print "Finished after step %d with delta-error = %f error = %f" % (step, de, error)
            break
    return w

def y(w, phi):
    """
    w:      vector with dimension M(K - 1)
    phi:    matrix with dimension N(K - 1) x M(K - 1)

    y:      vector with dimension N(K - 1)
    """
    expA = np.array(map(math.exp, np.dot(phi, w)))
    y = expA / expA.sum()
    return y

def grad_E(y, t, phi):
    """
    y:      vector with dimension N(K - 1)
    t:      vector with dimension N(K - 1)
    phi:    matrix with dimension N(K - 1) x M(K - 1)

    grad_E: vector with dimension M(K - 1)
    """
    return np.transpose(phi).dot(y - t)

def Hessian(y, phi, K):
    """
    y:      vector with dimension N(K - 1)
    phi:    matrix with dimension N(K - 1) x M(K - 1)

    H:      matrix with dimension M(K - 1) x M(K - 1)
    """
    N = len(y) / (K - 1)

    #create (K - 1) x (K - 1) blocks of N x N identity matrices as a mask
    i = np.identity(N)
    for _ in range(K - 2):
        i = np.concatenate((i, np.identity(N)), axis=0)
    I = i
    for _ in range(K - 2):
        I = np.concatenate((I, i), axis=1)
    #create R weighting matrix using mask from above
    R = ma.masked_array(y[:,None] * -y + y[:,None] * np.identity(len(y)), mask=1-I).filled(0)
    H = np.transpose(phi).dot(R).dot(phi)
    return H

def w_new(w, y, t, phi, K, alpha=1):
    return w - alpha * np.linalg.inv(Hessian(y, phi, K)).dot(grad_E(y, t, phi))

"""
N: # of inputs
M: # of features
K: # of classes
w (KxM): w_km is weights for kth class, mth feature
y (NxK): y_nk is posterior probability for nth input, kth class
phi (NxM): phi_nm is basis function for nth input, mth feature
t (NxK): output for nth input, kth class
Author: Sam Kim
"""
def logRegress(phi, t):
    w = initW(t.shape[1], phi.shape[1])
    while True:
        y = calcY(w,phi)
        hessMat = calcHessMat(y, phi)
        grad = calcGradE(y, t, phi)
        wNew = updateW(w, y, t, phi, hessMat, grad)
        if converged(w, wNew):
            break;
        w = wNew
    return wNew

def initW(k, m):
    #return np.zeros((k,m), dtype=float)
    return np.random.rand(k,m)

"""returns phi
no function is applied, but 1 is appended as a feature for the bias
"""
def basisNone(x):
    phi = np.insert(x, 0, 1.0, axis=1)
    return phi


"""converts t into a numpy array
example: t=[0,2,1], k is number of classes
output: tNew = [[1,0,0],[0,0,1],[0,1,0]
"""
def vectT(t, k=None):
    if k==None:
        k = np.amax(t)
    tNew = np.zeros((len(t), k))
    for m in range(len(t)):
        tNew[m,t[m]-1] = 1.0
    return tNew


"""
Calculates posterior probabilities,
y_k(phi)=exp(a_k)/Sum_j(exp(a_j))
where a_k = w_k * phi

DONE VECTORISING
"""
def calcY(w, phi):
    # NxK array of exp(a_k) for each n
    expA = np.exp(np.inner(phi,w))
    # N array of Sum_j(exp(a_j)) for each n
    sums = np.sum(expA, axis=1)
    y = expA / sums[:, np.newaxis]

    return y

"""Calculates the Hessian matrix
(K*M)x(K*M) where block j,k of size MxM given by sum_n...
TODO: Vectorize
"""
def calcHessMat(y, phi):
    m = phi.shape[1]
    N = y.shape[0]
    K = y.shape[1]
    hess = np.empty((K,K,m,m))
    i = np.identity(K)
    temp = np.empty((N,m,m))
    for k in range(K):
        for j in range(K):
            for n in range(N):
                temp[n] = y[n,k]*(i[k,j]-y[n,j])*np.outer(phi[n],phi[n])
            hess[j,k] = np.sum(temp, axis=0)

    hess = hess.swapaxes(1, 2).reshape(K*m, K*m)
    return hess

"""Gradient of the error function with respect to each of w_j
KxM
"""
def calcGradE(y, t, phi):
    grad = np.sum((y - t)[:,:,np.newaxis]*phi[:,np.newaxis,:], axis=0)
    return grad

"""Updates w based on the Newton-Raphson iterative optimization (IRLS)
"""
def updateW(w, y, t, phi, hessMat, gradE):
    return w - np.dot(np.linalg.pinv(hessMat), gradE)

"""Compares two vectors and tests if they are close enough to each other"""
def converged(wOld, wNew, rtol=1e-03, atol=1e-03):
    return allclose(wOld, wNew, rtol, atol)
