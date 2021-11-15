import math



import  numpy  as np
import sklearn
import matplotlib.pyplot as plt

import random
from sklearn import preprocessing, datasets, decomposition, metrics
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
X = iris.data
Y = iris.target


#k-means
def k_means(X,nbr_clusters):
    #choisir aléztoitement les clusters
    nbr=random.choice(X)
    clusters=[]

    for i in range(nbr_clusters):
            clusters.append(X[i,:])
    print("les  permiers clusters ")
    print(clusters)
    for iter in range(10):
        #etpae3 calcul des distances entre ces clusters et les valeurs de X
        distance = metrics.pairwise.euclidean_distances(X, clusters )
        #on affecte pour chaque individu le cluster le plus proche

        m=-1
        Y=[]
        for i in range(len(distance)):
            Y.append(np.argmin(distance[i,:]))
        #calcul des nouveau culsters ==> la moyenne de chaque classe
        #classes contient les nouveau clusers qui sont les moyenne de chaque classes
        clusters=[]
        np.array(clusters)
        moy=[]
        for i in range(0,nbr_clusters):
            clusters.append(np.mean(X[np.where(np.array(Y)==i)],axis=0))

        #for i ,k in  zip(range (0,nbr_clusters), range(len(Y))):
           # if Y[k]==i:
                #clusters.append(moy[i])




    print("les derniers clusters==>")
    return clusters



print("K-means")
print(k_means(X,2))






