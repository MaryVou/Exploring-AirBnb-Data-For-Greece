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

X_test = np.array(df_thes.drop(['price','city','month'],axis=1))
y_test = np.array(df_thes['price'])

print("\nTRAINING ON ATHENS DATA AND TESTING ON THESSALONIKI")
linear(X_train, y_train, X_test, y_test,"generalization_linear_1")
ridge(X_train, y_train, X_test, y_test,"generalization_ridge_1")
lasso(X_train, y_train, X_test, y_test,"generalization_lasso_1")
elasticnet(X_train, y_train, X_test, y_test,"generalization_elasticnet_1")
dtreeregressor(X_train, y_train, X_test, y_test,"generalization_dtreereg_1")
gradientboosting(X_train, y_train, X_test, y_test,"generalization_gradientboosting_1")
randomforest(X_train, y_train, X_test, y_test,"generalization_randomforest_1")

##########################################################
# 		  PART B: TRAIN ON JUNE AND TEST ON JULY         #
##########################################################

df_june = df.loc[df['month'] == 'June']
df_july = df.loc[df['month'] == 'July']

X_train = np.array(df_ath.drop(['price','city','month'],axis=1))
y_train = np.array(df_ath['price'])

X_test = np.array(df_thes.drop(['price','city','month'],axis=1))
y_test = np.array(df_thes['price'])

print("\nTRAINING ON JUNE DATA AND TESTING ON JULY")
linear(X_train, y_train, X_test, y_test,"generalization_linear_2")
ridge(X_train, y_train, X_test, y_test,"generalization_ridge_2")
lasso(X_train, y_train, X_test, y_test,"generalization_lasso_2")
elasticnet(X_train, y_train, X_test, y_test,"generalization_elasticnet_2")
dtreeregressor(X_train, y_train, X_test, y_test,"generalization_dtreereg_2")
gradientboosting(X_train, y_train, X_test, y_test,"generalization_gradientboosting_2")
randomforest(X_train, y_train, X_test, y_test,"generalization_randomforest_2")