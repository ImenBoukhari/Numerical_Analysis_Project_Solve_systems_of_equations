import numpy as np
import scipy.linalg as spl
from numpy.linalg import norm 
import time
from scipy.sparse.linalg import cg

class bcolors: #Affichage avec couleurs.
    OK = '\033[92m' #vert
    RG = '\033[91m' #rouge
    RESET = '\033[0m' #rest des couleurs

def symdefpos(A):
    # vérifier si M est symétrique (AT=A) &&  la matrice est définie positive
    if (np.array_equal(A, np.transpose(A))) and (  np.all(np.linalg.eigvals(A) > 0)) and (spl.det(A) != 0):
        return True
    else :
        return False


## gradient conjugé
def conjugue(A,b,X,itMax,tol):
    if (symdefpos(A) == False):
        print(bcolors.FAIL+"\n A n'est pas symétrique définie positive")
    else :
        R = b-A.dot(X) #-gradient de f(Xk)
        P = R # Direction initiale (-gradient de la fonction)
        k=0
        
        while (k<=itMax) and (norm(R) > tol):#verification des condition: #La précision fixée à 10e-5 
                                                #&& nombre d'itération ne dépasse pas nbr max
            Ap = A.dot(P) # A * P
            alpha = np.transpose(R).dot(R) / np.transpose(P).dot(Ap) #pas
            X = X + (alpha * P) # X(k+1) = X(k) + direction(k) * pas(k)
            Rancien=R #R(k) -->gradient f(k+1)
            R = R - (alpha * Ap) #R(k+1) --> -gradient f(k+1)

            beta = (np.transpose(R).dot(R)/np.transpose(Rancien).dot(Rancien))
            P = R + beta* P # direction k+1

            k=k+1   #incrémentation d'itération
        print("\nle nombre d'itération = \n",k)

        print(bcolors.OK+"\n notre solution minimale cherchée X = \n", X)

## fonction test
def test():
    choix = (int)(input("\nEntrer 1 si fonction test Entrer 2 si matrice aléatoire\n"))

    if(choix==1) :#fonction test
        b= np.array([[0.],[1.]])#b=vecteur colonne (0 1)
        X0 = np.array([[0.],[0.]]) #x0=vecteur colonne (0 0)
        A= np.array([[5.,4.],[4.,5.]])#A=matrice carré
        tol = 1e-5 #La précision fixée à 10e-5

        t1=time.time()#temps de départ
        conjugue(A, b, X0,100, tol)#exécution de la fonction
        t2=time.time()#temps de terminaison de l'execution de la fonction
        print(bcolors.RESET+"temps de calcul = ",t2-t1)#temps de calcul

    elif(choix==2):#matrice aléatoire
        n=int(input("\nEntrer taille du matrice\n"))
        mat = np.random.normal(size=[n,n])#construction d'une matrice carré ayant des valeurs aléatoires
        A = np.dot(mat, mat.transpose())#pour qu'on soit sure que A est def positive, inversible et symétrique 
        b = np.random.normal(size=[n,1])
        tol = 1e-5 #La précision fixée à 10e-5
        X0 = np.zeros((n,1))#initialisation de X0

        t1=time.time()#temps de départ
        itMax=int(input("\nEntrez le nombre max d'itération\n"))#nbr max d'itération
        conjugue(A, b, X0,itMax, tol)#exécution de la fonction
        t2=time.time()#temps de terminaison de l'execution de la fonction
        print(bcolors.RESET+"Le temps de calcul = ",t2-t1)#temps de calcul

        print(bcolors.RG+"\nvérification de notre solution\n")
        t3=time.time()#temps de départ de fonction prédefinie solve
        X_solve = spl.solve(A, b)
        print(bcolors.OK+"\n Avec la fonction solve X = \n", X_solve) 
        t4=time.time()#temps de terminaison de l'execution de la fonction prédefinie solve
        print(bcolors.RESET+"Temps de calcul de la fonction solve X = ",t4-t3)#temps de calcul 

        t5=time.time()#temps de départ de fonction prédefinie cg
        x3 = cg(A, b)#exécution de la fonction cg
        print(bcolors.OK+"\n Avec la fonction cg X = \n", x3) 
        t6=time.time()#temps de terminaison de l'execution de la fonction cg
        print(bcolors.RESET+"Temps de calcul de la fonction cg X = ",t6-t5)#temps de calcul 

    


if __name__ == '__main__':
    test()
        
