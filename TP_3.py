import math

import datasets as datasets
import  numpy  as np
import sklearn
import matplotlib.pyplot as plt
from fontTools.subset import subset

from sklearn import preprocessing, datasets, decomposition, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Plus Proche Voisin



"""

def PPV(X,Y):
   distance=metrics.pairwise.euclidean_distances(X,X)

#prendre le min
 
   label=[]
   var=0
   indice2=-1
   min = np.max(distance)
   for i in range (0,len(X)):
       for j in range (0,len(X)):

           if i!= j:
               if min>=distance[i][j]:
                   min=distance[i][j]
                   var=1
                   indice2=j

       if i!= j:
           label.append(Y[indice2])
       min = np.max(distance)










   return label;







def PPV2(X,Y):


    distance = metrics.pairwise.euclidean_distances(X, X)

    # prendre le min

    label = []

    indice2 = -1
    er=0
    min = np.max(distance)

    for i in range(0, len(X)):
        for j in range(0, len(X)):

            if i != j:

                if min >=distance[i][j]:
                    min = distance[i][j]

                    indice2 = j


        if  i != j:

            label.append(Y[indice2])

        min = np.max(distance)





    for i ,j in zip(label,Y):
        if(i!=j):
            er=er+1


    return  er*100/len(Y)







print("pourçentage d'erreur ")
print("nombre de label réel",len(Y))
print(PPV2(X,Y))


#utilisation k plus proche voisin de sklearn
#cette méthode ne prend pas en considération les valeurs 0 de la diagonal de la matrice de distance
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, Y)
classifier=neigh.predict(X)
print(len(classifier))
er=0
for i, j in zip(classifier, Y):
    if (i != j):
        er = er + 1

print("taux d'erreur pour le neighborsClassifier ",er*100/len(classifier))

#question bonus
def PPV3(X,Y,k):
   distance=metrics.pairwise.euclidean_distances(X,X)

#prendre le min

   label=[]
   var=0
   indice2=-1
   er=0
   min = np.max(distance)
   #k ligne
   for i in range (0,k):
       for j in range (0,len(X)):

           if i!= j:
               if min>=distance[i][j]:
                   min=distance[i][j]
                   var=1
                   indice2=j

       if i!= j:
           label.append(Y[indice2])
       min = np.max(distance)

   for i ,j in zip(label,Y):
        if(i!=j):
            er=er+1


   return  er*100/len(Y)

print("question bonus ==> k voisins ")
print(PPV3(X,Y,len(Y)))
"""
def CBN(X,Y):
          print(iris.target_names)
          #les donnes dont la classe est ==>0
          class0=X[:50,:]
          print(class0)
          class1=X[50:100,:]
          print(class1)
          class2=X[100:150,:]
          print(class2)
          centre0=np.mean(class0,axis=0)
          print("centre0==>",centre0)
          centre1=np.mean(class1,axis=0)
          print("centre1==>",centre1)
          centre2=np.mean(class2,axis=0)
          print("centre2==>",centre2)
          #calcule de la probabilité
          #ici les claasses sont equiprobable car on a le meme nombre d"individus pour chaque classe 50
          p0=len(class0)/len(X)
          print("probabilité de la classe0==>",p0)
          p1=len(class1)/len(X)
          print("probabilité de la classe1==>",p1)
          p2=len(class2)/len(X)
          print("probabilité de la classe2==>",p2)

          #clalcule de P(x/w) pour chaque classe==pc0,pc1,pc2
          #class0
          print("calcul de la probalité conditionnelle pour chaque element de la classe0==>")
          distance0=metrics.pairwise.euclidean_distances(X,[centre0])
          var=metrics.pairwise.euclidean_distances(X,[centre0,centre1,centre2])
          s=np.sum(var,axis=1)
          pc0=1-distance0.ravel()/s
          print("P(x/class0)==>",pc0)
          print(pc0.shape)
          print("distance entre chaque element x  de la classe0 et le centre0",distance0.ravel())
          #class1
          print("calcul de la probalité conditionnelle pour chaque element de la classe1==>")
          distance1=metrics.pairwise.euclidean_distances(X,[centre1])
          var=metrics.pairwise.euclidean_distances(X,[centre0,centre1,centre2])
          s=np.sum(var,axis=1)
          pc1=1-distance1.ravel()/s
          print("P(x/class0)==>",pc1)
          print(pc1.shape)
          print("distance entre chaque element x  de la classe0 et le centre0",distance1.ravel)
          #class2
          print("calcul de la probalité conditionnelle pour chaque element de la classe2==>")
          distance2=metrics.pairwise.euclidean_distances(X,[centre2])

          var=metrics.pairwise.euclidean_distances(X,[centre0,centre1,centre2])
          s=np.sum(var,axis=1)
          pc2=1-distance1.ravel()/s
          print("P(x/class0)==>",pc2)
          print(pc2.shape)
          print("distance entre chaque element x  de la classe0 et le centre0",distance2.ravel)
          #on cancatene tous les probabilité cond sdans un seul tableau que la fonction va renvoyer
          #prob_cond=np.concatenate((pc0,pc1,pc2),axis=0)
          #print("prob cond",prob_cond)
          #print(prob_cond*(p0))
          Y_predict=[]
          for i in range(0,len(pc0)):
                m=max(pc0[i],pc1[i],pc2[i])
                if m==pc0[i]:
                    Y_predict.append(0)
                else :
                       if m==pc1[i]:
                            Y_predict.append(1)
                       else:
                            if m==pc2[i]:
                                 Y_predict.append(2)

          er = 0

          print("leng",len(Y_predict))
          for i, j in zip(Y_predict, Y):
              if (i != j):
                  er = er + 1


          return er*100/len(Y_predict)

#ques†ion2
nbr_err=CBN(X,Y)
print("taux d'erreur pour le neighborsClassifier ", nbr_err,"%")
#question3
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
Y_pred=clf_pf.predict(X)
print(Y_pred)

