import pandas as pd
import numpy as np
from classification_models import dtree,kneighbors,probabilistic,rfc,svm


df = pd.read_csv("main_df.csv")
del df["Unnamed: 0"]

##########################################################
# 	PART A: TRAIN ON ATHENS AND TEST ON THESSALONIKI     #
##########################################################

df_ath = df.loc[df['city'] == 'Athens']
del df_ath['city']
del df_ath['month']

df_thes = df.loc[df['city'] == 'Thessaloniki']
del df_thes['city']
del df_thes['month']

df_ath['range'] =pd.cut(df_ath['price'], bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_ath = df_ath.drop(["price"],axis=1)

X_train = np.array(df_ath.drop(['range'],axis=1))
y_train = np.array(df_ath['range'])


df_thes['range'] = pd.cut(df_thes['price'],bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_thes = df_thes.drop(["price"],axis=1)

X_test = np.array(df_thes.drop(['range'],axis=1))
y_test = np.array(df_thes['range'])

print("\nTRAINING ON ATHENS DATA AND TESTING ON THESSALONIKI")

dtreeResult = rfc(X_train,y_train,X_test,y_test,"")

##########################################################
# 		  PART B: TRAIN ON JUNE AND TEST ON JULY         #
##########################################################

df_june = df.loc[df['month'] == 'June']
del df_june['city']
del df_june['month']

df_july = df.loc[df['month'] == 'July']
del df_july['city']
del df_july['month']

df_june['range'] =pd.cut(df_june['price'], bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_june = df_june.drop(["price"],axis=1)

X_train = np.array(df_june.drop(['range'],axis=1))
y_train = np.array(df_june['range'])


df_july['range'] = pd.cut(df_july['price'],bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_july = df_july.drop(["price"],axis=1)

X_test = np.array(df_july.drop(['range'],axis=1))
y_test = np.array(df_july['range'])

print("\nTRAINING ON JUNE DATA AND TESTING ON JULY")

dtreeResult = rfc(X_train,y_train,X_test,y_test,"")