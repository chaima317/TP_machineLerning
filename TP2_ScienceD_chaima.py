
import  numpy  as np
from lda import LDA
from sklearn import preprocessing, datasets, decomposition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#A. Normalisation de données
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

x=np.array([[ 1, -1, 2],[2, 0, 0],[0, 1, -1]])
print(x)
print("dimension de la matrice",x.shape)

#calcule de la moyenne et la variance
# normalisation==> https://datascientest.com/normalisation-des-donnees
moy=x.mean(0);
print("la moyenne de X",moy)
vari=x.var(0);
print("la variance X ",vari)

x= preprocessing.scale(x)
print("matrice normalisé de X ")
print(x)
print("Sa a moyenne",x.mean(0))
print("Sa a variance ",x.var(0))
print("l'ecart type",x.std(0))
#la normalisation d'une variable est le fait d'amener ses valeurs autour de 0
#le but aussi de la standarisation est d'amener( l'ecart type/variance) autour de 1
#la moyenne de chaque variable est autour de 0 et sa variance autour de 1

#B. Normalisation MinMax

x2=np.array([[1, -1, 2],[2, 0, 0],[0, 1, -1]])
print("matrice x2",x2)
print("dimension de la matrice",x2.shape)
print("la moyenne de X2 ",x2.mean(0))
print("la variance de X2 ",x2.var(0))

#normalisation
scaler=preprocessing.MinMaxScaler()
x2 = scaler.fit_transform(x2)

#print(scaler.fit(x2))
print("matrice normalisé de X2 ")
print(x2)
print("Sa a moyenne",x2.mean(0))
print("Sa a variance ",x2.var(0))
print("l'ecart type",x2.std(0))
#

#C. visualisation de données

iris = datasets.load_iris()

print(iris)
print(iris.data.shape)
var1=iris.data[:,0]
var2=iris.data[:,1]
plt.scatter(var1,var2,c=iris.target)

plt.xlabel("variable1")
plt.ylabel("variable2")
plt.title("Graphe")


plt.show()
#cette visualisation n'est pas tres claire les elements de deux classe sont mélangé ==> mauvaise classification

print(iris.data.shape)
var1=iris.data[:,0]
var3=iris.data[:,2]
plt.scatter(var1,var3,c=iris.target)

plt.xlabel("variable1")
plt.ylabel("variable3")
plt.title("Graphe")
plt.show()
#il reste toujours quelques  valeurs corélé

print(iris.data.shape)
var1=iris.data[:,0]
var4=iris.data[:,3]
plt.scatter(var1,var4,c=iris.target)

plt.xlabel("variable1")
plt.ylabel("variable4")
plt.title("Graphe")
plt.show()
#deux classe corélé entre eux ==> deux classes non déparées correctement 


print(iris.data.shape)
var2=iris.data[:,1]
var3=iris.data[:,2]
plt.scatter(var2,var3,c=iris.target)

plt.xlabel("variable2")
plt.ylabel("variable3")
plt.title("Graphe")
plt.show()

#meme probleme que  le précédent


print(iris.data.shape)
var3=iris.data[:,2]
var4=iris.data[:,3]
plt.scatter(var3,var4,c=iris.target)

plt.xlabel("variable2")
plt.ylabel("variable3")
plt.title("Graphe")
plt.show()

#https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales ==> trés bon exemple pour comprendre l'acp
#D. Réduction de dimensions et visualisation de données

print(iris.data.shape)
#on choisi les 4 variables de notre data==> n_coponents=4
pca = PCA(n_components=2)

irisPCA=pca.fit_transform(iris.data)
print("irisPCA")
print(irisPCA)

plt.scatter(irisPCA[:,0],irisPCA[:,1],c=iris.target)

plt.xlabel("composante1")
plt.ylabel("composante2")
plt.title("PCA ")
plt.show()

#lda
lda = LinearDiscriminantAnalysis(n_components=2)
irisLDA=lda.fit_transform(iris.data,iris.target)
print("irisLDA")
print(irisLDA)

plt.scatter(irisLDA[:,0],irisLDA[:,1],c=iris.target)
plt.xlabel("composante1")
plt.ylabel("composante2")
plt.title("lda ")
plt.show()
#on remarque que l'da est plus performante que la pca en termes de séparation de classent
#dans la PCA ==> le chauvauchement des classes est plus important
