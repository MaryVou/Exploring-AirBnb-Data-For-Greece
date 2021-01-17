import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

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

df = pd.read_csv("final_df1.csv")
#print(df.head())
#print(df.shape)


df['range'] = pd.cut(df.price, [0,50,100,200,500], include_lowest=True)
dfsub=df.drop('price',axis=1)
#sns.pairplot(dfsub, hue='range')
#plt.show()

from sklearn.model_selection import train_test_split
X=dfsub.drop('range',axis=1)
y=dfsub['range']
y=y.astype('str')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


#######################################################
#              Decision Tree Classifier               #
#######################################################
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print("#######################################################")
print("#              Decision Tree Classifier               #")
print("#######################################################")
print("Accuracy:  {}".format(accuracy_score(y_test, predictions)))
print("Matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
print("------------------  Report  ---------------------------")
print(classification_report(y_test,predictions))
dtreeResult = accuracy_score(y_test, predictions)
#######################################################
#                KNeighbors Classifier                #
#######################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("#######################################################")
print("#                KNeighbors Classifier                #")
print("#######################################################")
print("Accuracy:  {}".format(accuracy_score(y_test, predictions)))
print("Matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
print("------------------  Report  ---------------------------")
print(classification_report(y_test,predictions))
knnResult = accuracy_score(y_test, predictions)


#######################################################
#              Probabilistic Classifier               #
#######################################################
from sklearn.naive_bayes import GaussianNB
prob = GaussianNB()
prob.fit(X_train, y_train)
predictions = prob.predict(X_test)
print("#######################################################")
print("#              Probabilistic Classifier               #")
print("#######################################################")
print("Accuracy:  {}".format(accuracy_score(y_test, predictions)))
print("Matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
print("------------------  Report  ---------------------------")
print(classification_report(y_test,predictions))
probResult = accuracy_score(y_test, predictions)
#######################################################
#                   SVM Classifier                    #
#######################################################
from sklearn.svm import SVC
SVM = SVC()
SVM.fit(X_train,y_train)
predictions = SVM.predict(X_test)
print("#######################################################")
print("#                   SVM Classifier                    #")
print("#######################################################")
print("Accuracy:  {}".format(accuracy_score(y_test, predictions)))
print("Matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
print("------------------  Report  ---------------------------")
print(classification_report(y_test,predictions))
SVMResult = accuracy_score(y_test, predictions)

print("\n")
print("##################    Final Report    #################")
print("Decision Tree Classifier: {}".format(dtreeResult))
print("KNeighbors Classifier: {}".format(knnResult))

print("Probabilistic Classifier: {}".format(probResult))
print("SVM Classifier: {}".format(SVMResult))
print("#######################################################")

#print(df.groupby('range').size())
#df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()