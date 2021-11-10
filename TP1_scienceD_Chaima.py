from sklearn import datasets
from sklearn.datasets import fetch_openml, make_blobs
import matplotlib.pyplot as plt
import  numpy  as np
import  matplotlib.pyplot as py
#iris c'est uen matrice pas un dictionnaire
#ON A UNE MATRICE DE 150 LIGNE ET 4 VARIABLE
#POUR CHAQUE LIGNE ON A UN TARGET QUI EST AJOUTÉ (UN ENCODAGE 0,1 OU 2 ) QUI CORRESPOND RESPECTIVEMENT AU TARGET NAMES ['setosa' 'versicolor' 'virginica']
#ET ça cest valable pour toute les bases de donnes qu'on peut trouver dans datasets
#Manipulation d’un jeu de données


iris = datasets.load_iris()

print(iris)
print("nom des classe ")
print(iris.target_names)
print(" l'encodage de la classe pour chaque ligne ")
print(iris.target)
print("target names pour chaque ligne ")
print(iris.target_names[iris.target])

print("la moyenne pour chaque variable ==pour chaque colonne ")
print(iris.data.mean(0))
print("l'ecart type pour chaque variable ")
print(iris.data.std(0))
print("min chaque variable")
print(iris.data.min(0))
print("max pour chaque variable")
print(iris.data.max(0))

print("le nombre de donnes")
print(iris.__sizeof__())
print("le nombre de variable" )
#iris c'est un objet et iris. data c'est la matrice
print(iris.data.shape[1])

print("le nombre de ligne")
print(iris.data.shape[0])
print("le nombre de classes")
print(iris.target_names.shape)


#Téléchargement et importation de données



mnist = fetch_openml('mnist_784', as_frame=False)

print("affichage de la matrice de données")
print(mnist)
print("le nombre de données ")
print(mnist.__sizeof__())
print("le nombre de variable ")
print(iris.data.shape[0])
print("les numéro de classe pour chaque données")
print(iris.target)
print("moyenne pour chaque variable")
print(iris.data.mean(0))
print("l'cart type")
print(iris.data.std(0))

print("min")
print(iris.data.min(0))
print("max")
print(iris.data.max(0))

print("le nombre de classe ")
#unique ==> donnes la liste des valeurs existante dans le jeu de données
#exemple https://www.w3resource.com/numpy/manipulation/unique.php
print(np.unique(iris.target).shape)

#D. Génération de données et affichage

X, Y = make_blobs(n_samples=1000, centers=4, n_features=2,)

print(X.shape)
print(Y.shape)
plt.scatter(X[:,0],X[:,1] , c=Y)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Graphe")
plt.ylim((-15, 15))
plt.xlim((-15, 15))

plt.show()

A, B = make_blobs(n_samples=100, centers=2, n_features=2,)
C, D = make_blobs(n_samples=500, centers=3, n_features=2,)

#fusion  des données
E=np.vstack((A,C))
F=np.hstack((B,D))

plt.scatter(E[:,0],E[:,1] ,c=F)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Graphe")

plt.show()
#