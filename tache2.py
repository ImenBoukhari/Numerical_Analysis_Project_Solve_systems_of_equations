from sympy import exp
from sympy import symbols, diff, Symbol
from sympy.utilities.lambdify import lambdify
from sympy.solvers import solve
import numpy as np
import tkinter
from numpy import sin
from tkinter import simpledialog
def extract_symbols(string_func, nbS):
    i = 0
    s = ''
    while i < nbS :
        s += 'x'+str(i)+' '
        i+=1
    return symbols(s, real=True)
                    
    
def gradient(string_func, nbS):
    X = extract_symbols(string_func, nbS)
    func = eval(string_func)
    i = 0
    dX = [None] * nbS
    while i < nbS : 
        dX[i] = diff(func, X[i])
        i += 1
    return dX


    
def hessienne(dX,string_func):
    nbS = len(dX)
    i = 0
    X = extract_symbols(string_func, nbS)
    H = np.array([[None]* nbS, [None]* nbS]) 
    while i < nbS : 
        j = 0
        while j < nbS :
            H[i,j] = diff(dX[i], X[j])
            j += 1
        i += 1
    return H



def gradient_point(x, dX,string_func):
    nbS = len(dX)
    X = extract_symbols(string_func, nbS)
    surface_func = eval(string_func)
    i = 0
    dX = [None] * nbS
    while i < nbS : 
        dX[i] = diff(surface_func, X[i])
        i += 1
    res = np.array([])
    i = 0
    while i < nbS : 
        res = np.append(res, [lambdify(X[i], dX[i])(x[i])])
        i += 1
    return res


def task2(string_func):
	nbSymb = 2
	x = np.array([5, 2])
	dX = gradient(string_func, nbSymb)
	print(dX)
	print(gradient_point(x, dX,string_func))
	print(hessienne(dX,string_func))




