import numpy as np
import numdifftools as nd


#Definition de la fonction 2
def fct2(x):
    y = np.asarray(x)
    return (y[0]**2+y[1]**4)

def fct3(x):
    y = np.asarray(x)
    return np.sum((y[0] - 1)**2 + 100*(y[1] - y[0]**2)**2)


#Definition d'une fonction pour appliquer la methode de pas fixe.
def Gradient_Pas_Fixe(f,Tolerance,x0,pas,NB_ITR):
    #x contient les cordonnes du point initiale
    x = np.asarray(x0)
    #y utiliser pour stocker le point precedent pour la calcul de la precision
    y = x
    #definition de la foncion gradient du fonction f
    grad = nd.Gradient(f)
    for i in range(NB_ITR):
        #calcul du point pour chaque iteration avec methode du point fixe
        x = x - pas*grad(x)
        round_x = [round(num, 3) for num in x]
        #Variable contient la precision entre 2 points(courant et précédant)
        precision = np.linalg.norm(x - y)
        #Affichage des rèsultas chaque 100 itération
        #Stoper l'orsque en atteint la precision definit
        if (precision<Tolerance):
            break
        y = x
    #Affichage du résultat final
    print("Resultat pas fix:\n")
    print(f"x={round_x}\nIteration={i+1}\nf(x)={round(f(x),3)}")
    
    return { 'sol' : round_x, 'f' : round(f(x),3)}



x0 = np.array([1.1,2.1]) #defenition du point de depart
tolerance = 10**-5 #tolerance
itr = 10000
pas = 10**-3
#Appel de la fonction de calcul du gradient a pas fixe.
print("pour la fontion fct2, voici le resultat : it may take a few secondes ... please wait")
print("---------------------------------------------------------")
Gradient_Pas_Fixe(fct2,tolerance,x0,pas,itr)
print("---------------------------------------------------------")
print("pour la fontion fct3, voici le resultat :")
print("---------------------------------------------------------")
Gradient_Pas_Fixe(fct3,tolerance,x0,pas,itr)
print("-------------The end thank you for waiting --------------")

