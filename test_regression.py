import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from regression_models import linear, ridge, lasso, elasticnet, dtreeregressor, gradientboosting, randomforest


df = pd.read_csv("final_df.csv")
del df['Unnamed: 0']

##########################################################
# 	PART A: TRAIN ON ATHENS AND TEST ON THESSALONIKI     #
##########################################################

df_ath = df.loc[df['city'] == 'Athens']
df_thes = df.loc[df['city'] == 'Thessaloniki']

X_train = np.array(df_ath.drop(['price','city','month'],axis=1))
y_train = np.array(df_ath['price'])

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1, shuffle=True)

p_test = int(len(X_train)*0.1)

X_test = np.array(df_thes.drop(['price','city','month'],axis=1))
y_test = np.array(df_thes['price'])

X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1, shuffle=True)

X_test = X_test[:p_test]
y_test = y_test[:p_test]

print("\nTRAINING ON ATHENS DATA AND TESTING ON THESSALONIKI")
randomforest(X_train, y_train, X_test, y_test,"gen_reg_randomforest_1")

##########################################################
# 		  PART B: TRAIN ON JUNE AND TEST ON JULY         #
##########################################################

df_june = df.loc[df['month'] == 'June']
df_july = df.loc[df['month'] == 'July']

X_train = np.array(df_june.drop(['price','city','month'],axis=1))
y_train = np.array(df_june['price'])

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1, shuffle=True)

p_test = int(len(X_train)*0.1)

X_test = np.array(df_july.drop(['price','city','month'],axis=1))
y_test = np.array(df_july['price'])

X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1, shuffle=True)

X_test = X_test[:p_test]
y_test = y_test[:p_test]

print("\nTRAINING ON JUNE DATA AND TESTING ON JULY")
randomforest(X_train, y_train, X_test, y_test,"gen_reg_randomforest_2")