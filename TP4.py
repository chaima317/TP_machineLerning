import math



import  numpy  as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn import preprocessing, datasets, decomposition, metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
X = iris.data
Y = iris.target


#k-means
"""
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






    print("les derniers clusters==>")
    return clusters



print("K-means")
k=k_means(X,2)
print(k)
print("indice de silouhette")
print(sklearn.metrics.silhouette_score(X,k))
"""
#question2
"""
def k_means2(X, nbr_clusters):
    # choisir aléztoitement les clusters
    nbr = random.choice(X)
    buffer=[]
    clusters = []
    sil=[]

    for i in range(nbr_clusters):
        clusters.append(X[i, :])
    print("les  permiers clusters ")
    print(clusters)
    for iter in range(10):
        # etpae3 calcul des distances entre ces clusters et les valeurs de X
        distance = metrics.pairwise.euclidean_distances(X, clusters)
        # on affecte pour chaque individu le cluster le plus proche

        m = -1
        classes= []
        for i in range(len(distance)):
            classes.append(np.argmin(distance[i, :]))
        # calcul des nouveau culsters ==> la moyenne de chaque classe
        # classes contient les nouveau clusers qui sont les moyenne de chaque classes
        clusters = []
        np.array(clusters)
        moy = []
        for i in range(0, nbr_clusters):
            clusters.append(np.mean(X[np.where(np.array(classes) == i)], axis=0))
        buffer.append(clusters)
        print("sscore de silhouette , iteration",iter,"==>")
        sil.append(sklearn.metrics.silhouette_score(X,classes))
        print(sil)

    pca = PCA(n_components=2)
    XPCA = pca.fit_transform(X)
    #visualisation
    plt.scatter(XPCA[:, 0], XPCA[:, 1], c=classes)
    plt.xlabel("composante1")
    plt.ylabel("composante2")
    plt.title("PCA ")
    plt.show()
    print("Buffer des clusters==>",buffer)
    print("les silhouette",sil)
    m_sol=np.argmax(sil)
    print("le meileur  cluster==>")
    return buffer[m_sol]

print("K-means")
print(k_means2(X,2))

#essaie avec different nombre de clusters



for i in range(2,10):
    print("k means avec nbr de clusters", i)
    target=k_means2(X,i)
    print()

"""

#question4 ==>ACP==> fait
df = pd.read_csv ('choixprojetstab.csv',sep=";")
etu=df.to_numpy()
print(etu)
print(etu.shape)
Y=etu[:,1]
print("labels==>")
print(Y)
X=etu[1:,1:]
print("data==>")
print(X)
#mean-shift clustering algorithm¶
clustering = MeanShift(bandwidth=2).fit(X)
print("labels==>",clustering.labels_)







