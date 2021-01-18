import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classification_models import dtree,kneighbors,probabilistic,randomforest,svm

df = pd.read_csv("final_df.csv")
del df["Unnamed: 0"], df["city"], df["month"]


df['range'] = pd.cut(df.price, bins=[0,50,100,200,500], labels=["Less than 50","50-100","100-200","More than 200"],include_lowest=True)
df = df.drop(['price'],axis=1)
#sns.pairplot(dfsub, hue='range')
#plt.show()

X = np.array(df.drop(['range'],axis=1))
y = np.array(df["range"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,shuffle=True)

dtreeResult = dtree(X_train,y_train,X_test,y_test,"class_dtree")
knnResult = kneighbors(X_train,y_train,X_test,y_test,"class_kneighbors")
probResult = probabilistic(X_train,y_train,X_test,y_test,"class_prob")
SVMResult = svm(X_train,y_train,X_test,y_test,"class_svm")
RandomForestResult = randomforest(X_train,y_train,X_test,y_test,"class_randomforest")

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