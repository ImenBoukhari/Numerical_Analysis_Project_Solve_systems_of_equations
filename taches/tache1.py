#!/usr/bin/env python
# coding: utf-8

# In[24]:

from tkinter import * 
from tkinter.messagebox import *
import tkinter
from tkinter import simpledialog
import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin,cos,exp
import matplotlib.cm as cm
#*********task1 By Hana , Sofien and Haifa ************     
def graph_niveau():
    #Tracer le graphe
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)
    fn =  simpledialog.askstring("Input","fonction")
    print(eval(fn))
    Z = eval(fn)
    fig = plt.figure()
    ax = fig.gca(projection='3d') 
    print(cos(10)+sin(10))
    #ax.plot_surface(x, y, Z)
    ax.plot_surface(x, y, Z, cmap=cm.nipy_spectral_r)
    msg = "Graphe du fonction  " + fn
    plt.title(msg)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Z')
    #plt.legend("Graphe du fontion")
    plt.figure(2)
    plt.axis('equal')
    #plt.contourf(x, y, Z, 20)
    plt.contour(x, y, Z, cmap=cm.nipy_spectral_r)
    plt.colorbar()
    plt.show()
    #Tracer les Lignes de niveaux





# In[ ]:




