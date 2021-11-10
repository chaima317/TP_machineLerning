import math


import datasets as datasets
import  numpy  as np
import sklearn
import matplotlib.pyplot as plt
from fontTools.subset import subset
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
    #etape2
    distances=[]
    #etpae3 calcul des distances entre ces clusters et les valeurs de X
    distance = metrics.pairwise.euclidean_distances(X, clusters)
    y_predict=[]
    return distance


print("K-means")
print(k_means(X,2).shape)







