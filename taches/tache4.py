import math as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# La fonction Rosenbrock
def Rosenbrock(X):
    return ((1-X[0])** 2 +100 *((X[1] -X[0]**2) **2))


def norme(u):
    x = u[0]
    y = u[1]
    return mp.sqrt(x ** 2 + y ** 2)

def gradient_f(X):
    return ([- 2+ 2 * X[0]  + 400 * X[0] ** 3 - 400 * X[0] * X[1], -200 * X[0] ** 2 + 200 * X[1]])

# Gradient a pas optimal

def Omega(X, alpha):
    return ([X[0] - alpha * gradient_f(X)[0], X[1] - alpha * gradient_f(X)[1]])


def SectionDoree(X0, a, b, precision):  # Recherche du pas optimal
    tau = (1 + np.sqrt(5)) / 2
    it = 0
    err = b - a
    while np.abs(err) > precision:
        aprime = a + (b - a) / (tau * tau)
        bprime = a + (b - a) / tau
        c, d = Omega(X0, aprime), Omega(X0, bprime)
        if c > d:
            a = aprime
        elif c < d:
            b = bprime
        else:
            a, b = aprime, bprime

        err = b - a
        it = it + 1
    return (a + b) / 2
#output: Le nombre qui donne une estimation d’un minimum local de f avec une erreur plus petite que tol

alpha = SectionDoree([0, 1], 0, 1, 0.00001)

def PasOptimal(X0, alpha):
    x = X0[0]
    y = X0[1]
    listeXY = [[x, y]]
    k = 0
    kmax = 10000

    while (norme(gradient_f([x, y])) > 0.00001) and (k < kmax):
        X = [x, y]
        x, y = Omega(X, alpha)
        k += 1
        listeXY.append([x, y])
        X = [x, y]
        a = X0[0] - alpha * gradient_f(X0)[0]
        b = X0[1] - alpha * gradient_f(X0)[1]

    Resp = Rosenbrock([a, b])
    return (X, k, norme(gradient_f([x, y])), listeXY, Resp)




