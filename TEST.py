import  numpy  as np
from matplotlib import pyplot

from sklearn import preprocessing, datasets, decomposition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
X = iris.data
Y = iris.target
Y_pred=[]
err=[]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
for i in range(1,100):
       classifier = KNeighborsClassifier(n_neighbors=i)
       classifier.fit(X_train, y_train)
       Y_pred.append(classifier.predict(X_test))
       err.append(1-accuracy_score(y_test,Y_pred[i-1]))

pyplot.plot(range(1,100),err, color = 'red')

pyplot.xlim(0, 100)
pyplot.xlabel("k")
pyplot.ylabel("taux derreur")
pyplot.ylim(0,1)
pyplot.title('choix de k en fonction du taux derreur')
pyplot.show()

