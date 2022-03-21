from tkinter import * 
from tkinter.messagebox import *
import tkinter
from tkinter import simpledialog
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import *
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, Symbol 
# import main pour recupérer les panels windows
from taches.tache1 import *
from taches.tache2 import task2
from taches.tache8 import rosenbrock, rosenbrock_grad
from taches.tache33 import Gradient_Pas_Fixe,steepest_descent_armijo
from math import cos
import numdifftools as nd
import runpy
ws=Tk()
ws.title("Optimisation")
ws.geometry("1000x320")

#****************** input functions 
def extract_symbols(string_func, nbS):
    i = 0
    s = ''
    while i < nbS :
        s += 'x'+str(i)+' '
        i+=1
    return symbols(s, real=True)

def entree():
    #instance les variables pour le fonctionnement de eval
    nbvariable=simpledialog.askinteger(title="nbr variables",prompt="Donner le nombre de variables :")
    string_func= simpledialog.askstring(title="fonction",prompt="Entrer votre fonction :")
    X = extract_symbols(string_func, nbvariable)
    func = eval(string_func)
    return func
#******************* implémentations des bouttons 
def entree_r2_b1(): 
    graph_niveau()
    
def entree_r2_b2():
    fn =  simpledialog.askstring("Input","fonction") 
    task2(fn)

def entree_r2_b3():
	f = simpledialog.askstring("Input","fonction") 
	x0 = np.array([1.1,2.1]) #defenition du point de depart
	tolerance = 10**-5 #tolerance
	itr = 10000
	pas = 10**-3
	Gradient_Pas_Fixe(f,Tolerance,x0,pas,NB_ITR)
    
def entree_r2_b4(): 
    fenetre = Toplevel()		  # Popup 
    #le menu
    #chaque bouton donne un 2ème choix : fonc prédifine ou nouvelle fonc
    fenetre.title('Choix algorithme')
    Label(fenetre, text='Les différents algorithmes à appliquer',background="#34A2FE").pack(padx=10, pady=10)
    
    Button(fenetre, text='Méthode de gradient à pas fixe', command=choixTypeFnGradientFixe).pack(padx=2, pady=2)
    Button(fenetre, text='Méthode de gradient à pas optimal', command=choixTypeFnGradientOpt).pack(padx=2, pady=2)
    Button(fenetre, text='Méthode de gradient conjugé standard', command=predefConjStand).pack(padx=2, pady=2)
    Button(fenetre, text="Le calcul du pas d'Armijo", command=choixArmijo).pack(padx=2, pady=2)
    Button(fenetre, text="Le calcul du pas de Wolfe", command=predefWolfe).pack(padx=2, pady=2)
    Button(fenetre, text="Méthode de gradient conjugé pour les fonctions non linéaires, pas de Wolfe", command=predefWolfeNonL).pack(padx=2, pady=2)
    #configuration du popup
    fenetre.grab_set()		  # Interaction avec fenetre jeu impossible
        #choisir entre fonction prédifine ou fonctions au choix (pour btn 4)
def choixTypeFnGradientOpt():
    fenetre2 = Toplevel()		  # Popup 
    #le menu
    #chaque bouton donne un 2ème choix : fonc prédifine ou nouvelle fonc
    fenetre2.title('Choix Fonctions')
    Label(fenetre2, text='Voulez-vous utiliser les fonctions prédifinies ou entrer une nouvelle ?',background="#34A2FE").pack(padx=10, pady=10)
    
    Button(fenetre2, text='Fonctions prédifinies', command=tache4prdef).pack(padx=2, pady=2)
    Button(fenetre2, text='Entrer une fonctions', command=tache4).pack(padx=2, pady=2)
    
    #configuration du popup
    fenetre2.grab_set()		  # Interaction avec fenetre jeu impossible
    ws.wait_window(fenetre2)   # Arrêt script principal
def tache4prdef():
    runpy.run_path("./taches/task4_fct2.py")
    runpy.run_path("./taches/task4_fct3rosenbrock.py")
def tache4():
    fn =  simpledialog.askstring("Input","fonction") 
    

def choixTypeFnGradientFixe():    
    fenetre2 = Toplevel()		  # Popup 
    #le menu
    #chaque bouton donne un 2ème choix : fonc prédifine ou nouvelle fonc
    fenetre2.title('Choix Fonctions')
    Label(fenetre2, text='Voulez-vous utiliser les fonctions prédifinies ou entrer une nouvelle ?',background="#34A2FE").pack(padx=10, pady=10)
    
    Button(fenetre2, text='Fonctions prédifinies', command=tacheGradientpasfixe).pack(padx=2, pady=2)
    Button(fenetre2, text='Entrer une fonctions', command=tacheGradientpasfixeWithNewFn).pack(padx=2, pady=2)
    
    #configuration du popup
    fenetre2.grab_set()		  # Interaction avec fenetre jeu impossible
    ws.wait_window(fenetre2)   # Arrêt script principal
    
def tacheGradientpasfixe():
    runpy.run_path("./taches/tache3.py")
    
def tacheGradientpasfixeWithNewFn():
    fn =  simpledialog.askstring("Input","fonction") 
    X=extract_symbols(fn,2)
    NB_ITR = simpledialog.askinteger("Input","nbr d'itérations")
    pas = simpledialog.askinteger("Input","le pas")
    x0 = np.array([1.1,2.1]) 
    fn=eval(fn)
    Gradient_Pas_Fixe(fn,10**-5,x0,pas,NB_ITR)

def predefConjStand(): 
    runpy.run_path("./taches/tache5.py")
    

def entree_r2_b5():
    runpy.run_path("./taches/tache9.py")
def choixArmijo():
    fenetre2 = Toplevel()		  # Popup 
    #le menu
    #chaque bouton donne un 2ème choix : fonc prédifine ou nouvelle fonc
    fenetre2.title('Choix Fonctions')
    Label(fenetre2, text='Voulez-vous utiliser les fonctions prédifinies ou entrer une nouvelle ?',background="#34A2FE").pack(padx=10, pady=10)
    
    Button(fenetre2, text='Fonctions prédifinies', command=predefArmijo).pack(padx=2, pady=2)
    Button(fenetre2, text='Entrer une fonctions', command=NonpredefArmijo).pack(padx=2, pady=2)
    
    #configuration du popup
    fenetre2.grab_set()		  # Interaction avec fenetre jeu impossible
    ws.wait_window(fenetre2)   # Arrêt script principal  
def NonpredefArmijo():
    nbS =  simpledialog.askinteger("Input","nbr de variables dans la fonction") 
    fn =  simpledialog.askstring("Input","fonction") 
    iterations =  simpledialog.askinteger("Input","nbr itérations")  
    X = extract_symbols(fn, nbS)
    func = eval(fn)
    grad = rosenbrock_grad(func)
    steepest_descent_armijo(func, grad, iterations)
def predefArmijo():
    runpy.run_path("./taches/tache6.py")
def predefWolfe():
    runpy.run_path("./taches/wolf.py")
def predefWolfeNonL():
    fn =  simpledialog.askstring("Input","fonction") 
    task2(fn) 
    



    
    
    #*************** main ***************************    
# main panel 
w1 = PanedWindow()  
w1.pack(fill = BOTH, expand = 1) 
# left panel
w2 = PanedWindow(w1, orient = VERTICAL,bd=20)
w1.add(w2)  
  #R2
r2_label=Label(w2, text="De R^2 vers R", background="#34A2FE") 

w2.add(r2_label)
r2_button1=Button(w2, text="Tracer le graphe de la fonction et les lignes de niveaux", command=entree_r2_b1)


w2.add(r2_button1) 
r2_button3=Button(w2, text="Afficher le vecteur gradient et la matrice hessienne de f ",command=entree_r2_b2)  

w2.add(r2_button3) 
r2_button4=Button(w2, text="Propser les différents algorithmes à appliquer avec les différents pas possibles", command=entree_r2_b4)  

w2.add(r2_button4)
r2_button5=Button(w2, text="Propser un comparatif de toutes les méthodes", command=entree_r2_b5)  


w2.add(r2_button5) 
#Rn
rn_label=Label(w2, text="R^n pour n>3", background="#34A2FE") 
w2.add(rn_label) 
r2_button1=Button(w2, text="Affiche le vecteur gradient et la matrice hessienne de f", command=entree_r2_b1)  
w2.add(r2_button1) 
r2_button1=Button(w2, text="Propser les différents algorithmes à appliquer avec les différents pas possibles", command=entree_r2_b1)  
w2.add(r2_button1) 
r2_button1=Button(w2, text="Propser un comparatif de toutes les méthodes", command=entree_r2_b1)  
w2.add(r2_button1) 

#right panel
w2 = PanedWindow(w1,orient = HORIZONTAL)  
w1.add(w2)  
right = Entry(w2, bd = 10)  
w2.add(right)   
   #*************end main *******************


ws.mainloop()



