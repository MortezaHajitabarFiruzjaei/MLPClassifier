###########################################################################################################################
###################################### Programmer: MEng. Morteza Hajitabar Firuzjaei ######################################
###########################################################################################################################
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X = iris.data
y = iris.target

MLP = MLPClassifier()
# MLP = Perceptron()
MLPClassifier(hidden_layer_sizes=(15,), random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

MLP.fit(X_train, y_train)
y_pred = MLP.predict(X_test)

scores = cross_val_score(MLP, X, y, cv=10)

print("Accuracy: %0.2f " % (scores.mean()))
