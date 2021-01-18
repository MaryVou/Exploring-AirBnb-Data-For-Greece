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

df = pd.read_csv("final_df.csv")
del df["Unnamed: 0"]
#print(df.head())
#print(df.shape)

df_ath = df[df['city'] == 'Athens']
del df_ath['city']
del df_ath['month']

df_thes = df[df['city'] == 'Thessaloniki']
del df_thes['city']
del df_thes['month']
#X = np.asarray(df_ath[["host_is_superhost","room_type","accommodates","month","deviation_from_mean_price","dist_from_center"]])
#Y = np.asarray(df_ath["price"])

df_ath['range'] =pd.cut(df_ath['price'], bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)
dfsub_ath= df_ath.drop('price',axis=1)
dfsub_ath['range'] = dfsub_ath['range'].astype("category").cat.codes
df_thes['range'] = pd.cut(df_thes['price'],bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)
dfsub_thes= df_thes.drop('price',axis=1)
dfsub_thes['range'] = dfsub_thes['range'].astype("category").cat.codes
#sns.pairplot(dfsub, hue='range')
#plt.show()

X_train = np.array(dfsub_ath)
y_train = dfsub_ath['range']
X_test = np.array(dfsub_thes)
y_test = dfsub_thes['range']
#from sklearn.model_selection import train_test_split
#X=dfsub.drop('range',axis=1)
#y=dfsub['range']
#y=y.astype('str')
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


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
#               Random Forest Classifier              #
#######################################################
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
RandomForest.fit(X_train,y_train)
predictions = RandomForest.predict(X_test)
print("#######################################################")
print("#               Random Forest Classifier              #")
print("#######################################################")
print("Accuracy:  {}".format(accuracy_score(y_test, predictions)))
print("Matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
print("------------------  Report  ---------------------------")
print(classification_report(y_test,predictions))
RandomForestResult = accuracy_score(y_test, predictions)

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
print("Random Forst Classifier: {}".format(RandomForestResult))
print("#######################################################")

#print(df.groupby('range').size())
#df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()