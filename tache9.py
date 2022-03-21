import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from warnings import warn
from sklearn.datasets import make_spd_matrix


    #il nous manque la tache 5 et 2 (tache 7 évaluer avec 8)
#import des taches 
import taches.tache3 as t3
import taches.task4_fct2 as t4
import time
import numpy as np
import taches.tache6 as t6
import taches.tache8 as t8
    
    #Tache 9 : collecte des outputs
def analyse_t3_fixe():
    x0 = np.array([1.1,2.1])
    error_point = 10**-10
    error_grad = 10**-10
    h = 10**-3
    iterations = 10000

    start = time.time()
    fct3 = t3.rosenbrock
    grad3 = t3.rosenbrock_grad
    result = t3.steepest_descent_constant_step(fct3, grad3, x0, iterations, error_point, error_grad, h)
    end = time.time()
    elapsed = end - start
    print(f'Méthode de Gradient conjugué à pas fixe : Temps d\'exécution : {elapsed:.2}ms, nombre d\'itération : {iterations} , X0 : {x0}')

def analyse_t3_variable():
    x0 = np.array([1.1,2.1])
    error_point = 10**-10
    error_grad = 10**-10
    h = 10**-3
    iterations = 10000
    start=time.time()
    fct2 = t3.f2
    grad2 = t3.f2_grad
    result = t3.steepest_descent_constant_step(fct2, grad2, x0, iterations, error_point, error_grad, h)
    end = time.time()
    elapsed = end - start

    print(f'Méthode de Gradient conjugué à pas variable décroissant : Temps d\'exécution : {elapsed:.2}ms, nombre d\'itération : {iterations} , X0 : {x0}')

    
def analyse_t6():
    f = t6.rosenbrock
    grad = t6.rosenbrock_grad
    x0 = np.array([1.1,2.1])
    error_point = 10**-10
    error_grad = 10**-10
    h = 10**-2
    iterations = 10000
    start = time.time()
    result = t6.steepest_descent_armijo(f, grad, x0, iterations, error_point, error_grad)
    end = time.time()
    elapsed = end - start
    print(f'Méthode de Gradient conjugué à pas Armijo : Temps d\'exécution : {elapsed:.2}ms, nombre d\'itération : {iterations} , X0 : {x0}')
    
def analyse_t4():
    alpha = t4.SectionDoree([0, 1], 0, 1, 0.00001)
    start = time.time()
    PO = t4.PasOptimal([0, 1], alpha)
    end = time.time()
    elapsed = end - start
    print(f'Méthode de Gradient conjugué à pas optimal : Temps d\'exécution : {elapsed:.2}ms, nombre d\'itération : {PO[1]} , X0 : {PO[0]}')


    
def analyse_t8():
    x0 = np.array([2, 1, 3])
    error_point = 10**-10
    error_grad = 10**-10
    h = 10**-2
    iterations = 10000
    start = time.time()
    
    x, num = t8.NonlinearCG(t8.Griewank, t8.GriewankGrad, init=x0)
    end = time.time()
    elapsed = end - start
    print(f'Méthode de Gradient conjugué pour les fonctions non linéaire : Temps d\'exécution : {elapsed:.2}ms, nombre d\'itération : {num} , X0 : {x}')

analyse_t3_fixe()
analyse_t3_variable()
analyse_t6()
analyse_t4()
analyse_t8()
