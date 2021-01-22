import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

################################################
#                 Functions 		           #
################################################

def printMetrics(y_test, predictions, model_name, plot_name):
	print("#######################################################")
	print("\n"+model_name)
	print("Accuracy:  {}".format(accuracy_score(y_test, predictions)))
	print("Matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
	print("------------------  Report  ---------------------------")
	print(classification_report(y_test,predictions))
	result = accuracy_score(y_test, predictions)
	return result 

################################################
#               	Models 		               #
################################################

def dtree(X_train, y_train, X_test, y_test,plot_name):
	dtree = DecisionTreeClassifier(criterion='entropy')
	dtree.fit(X_train,y_train)
	predictions = dtree.predict(X_test)

	result = printMetrics(y_test,predictions,"Decision Tree Classifier",plot_name)
	return result

def kneighbors(X_train, y_train, X_test, y_test,plot_name):
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	predictions = knn.predict(X_test)

	result = printMetrics(y_test,predictions,"KNeighbors Classifier",plot_name)
	return result

def probabilistic(X_train, y_train, X_test, y_test,plot_name):
	prob = GaussianNB()
	prob.fit(X_train, y_train)
	predictions = prob.predict(X_test)

	result = printMetrics(y_test,predictions,"Probabilistic Classifier",plot_name)
	return result

def rfc(X_train, y_train, X_test, y_test,plot_name):
	RandomForest = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
	RandomForest.fit(X_train,y_train)
	predictions = RandomForest.predict(X_test)

	result = printMetrics(y_test,predictions,"Random Forest Classifier",plot_name)
	return result

def svm(X_train, y_train, X_test, y_test,plot_name):
	SVM = SVC()
	SVM.fit(X_train,y_train)
	predictions = SVM.predict(X_test)

	result = printMetrics(y_test,predictions,"SVM Classifier",plot_name)
	return result