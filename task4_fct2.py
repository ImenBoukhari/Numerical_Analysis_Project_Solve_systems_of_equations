import math as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# La fonction 2
def fct2(X):
    return ((X[0])** 2 + (X[1])** 4)

def norme(u):
    x = u[0]
    y = u[1]
    return mp.sqrt(x ** 2 + y ** 2)

def gradient_f(X):
    return (2 * X[0], (4 * X[1])** 3)

# Gradient a pas optimal
def Omega(X, rho):
    return ([X[0] - rho * gradient_f(X)[0], X[1] - rho * gradient_f(X)[1]])

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

rho = SectionDoree([-1, -1], 0, 1, 0.01)
print (" rho :",rho)

def PasOptimal(X0, rho):
    x = X0[0]
    y = X0[1]
    listeXY = [[x, y]]
    k = 0
    kmax = 1000

    while (norme(gradient_f([x, y])) > 0.01) and (k < kmax):
        X = [x, y]
        x, y = Omega(X, rho)
        k += 1

        listeXY.append([x, y])#pour le graphe
        X = [x, y]
        a = X0[0] - rho * gradient_f(X0)[0]
        b = X0[1] - rho * gradient_f(X0)[1]

    Resp = fct2([a, b])

    return (X, k, norme(gradient_f([x, y])), listeXY, Resp)

PO = PasOptimal([-1, -1], rho)
print(" PasOptimal : [x,y] = ", PO[0], " \n et k = ", PO[1], "\n et Epsilon = ", PO[2])

# Affichage des courbes des iteres pour les 2 methodes
def CreerlistepointsX(liste):
    A = []
    for i in range(len(liste)):
        A.append(liste[i][0])

    return (A)

def CreerlistepointsY(liste):
    B = []
    for i in range(len(liste)):
        B.append(liste[i][1])

    return (B)

# Création de la liste contenant les valeurs de X et Y
C = CreerlistepointsX(PasOptimal([-1, -1], rho)[3])
D = CreerlistepointsY(PasOptimal([-1, -1], rho)[3])

plt.plot(C, D, "o", color="g", label=" Gradient à pas optimal ")
plt.legend()
plt.title(" Convergence des solutions ")
plt.xlabel(" X ")
plt.ylabel(" Y ")

ax = Axes3D(plt.figure())
X = np.linspace(-10, 10, 50)
Y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(X, Y)
Z = fct2([X, Y])
ax.plot_surface(X, Y, Z)
plt.title(" Fonction 2 ")
plt.xlabel(" X ")
plt.ylabel(" Y ")
# ax.plot(A,B,'o',color='r', label = " Gradient à pas fixe ")
# ax.plot(C,D,'o',color='y', label = " Gradient à pas optimal ")
plt.legend()
plt.show()  # affichage de la nappe
