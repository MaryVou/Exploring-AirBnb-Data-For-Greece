import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from regression_models import linear, ridge, lasso, elasticnet, dtreeregressor, gradientboosting, randomforest

##########################################################
# 	PART A: TEST DIFFERENT MODELS ON THE WHOLE DATASET   #
##########################################################

df = pd.read_csv("final_df.csv")

del df["Unnamed: 0"]

X = np.array(df.drop(['price','city','month'],axis=1))
Y = np.array(df["price"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,shuffle=True)

print("\nTRAINING AND TESTING ON THE WHOLE DATASET")
linear(X_train, y_train, X_test, y_test,"reg1_linear")
ridge(X_train, y_train, X_test, y_test,"reg1_ridge")
lasso(X_train, y_train, X_test, y_test,"reg1_lasso")
elasticnet(X_train, y_train, X_test, y_test,"reg1_elasticnet")
dtreeregressor(X_train, y_train, X_test, y_test,"reg1_dtreeregressor")
gradientboosting(X_train, y_train, X_test, y_test,"reg1_gradientboostingreg")
randomforest(X_train, y_train, X_test, y_test,"reg1_randomforestreg")

##########################################################
# 	PART B: TEST DIFFERENT MODELS ON DIFFERENT CITIES    #
##########################################################

df_ath = df.loc[df['city'] == 'Athens']

X = np.array(df_ath.drop(['price','city','month'],axis=1))
Y = np.array(df_ath['price'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

print("\nTRAINING AND TESTING ON DATA FROM ATHENS")
linear(X_train, y_train, X_test, y_test,"reg2_ath_linear")
ridge(X_train, y_train, X_test, y_test,"reg2_ath_ridge")
lasso(X_train, y_train, X_test, y_test,"reg2_ath_lasso")
elasticnet(X_train, y_train, X_test, y_test,"reg2_ath_elasticnet")
dtreeregressor(X_train, y_train, X_test, y_test,"reg2_ath_dtreeregressor")
gradientboosting(X_train, y_train, X_test, y_test,"reg2_ath_gradientboostingreg")
randomforest(X_train, y_train, X_test, y_test,"reg2_ath_randomforestreg")

###########################################################

df_thes = df.loc[df['city'] == 'Thessaloniki']

X = np.array(df_thes.drop(['price','city','month'],axis=1))
Y = np.array(df_thes['price'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,shuffle=True)

print("\nTRAINING AND TESTING ON DATA FROM THESSALONIKI")
linear(X_train, y_train, X_test, y_test,"reg2_thes_linear")
ridge(X_train, y_train, X_test, y_test,"reg2_thes_ridge")
lasso(X_train, y_train, X_test, y_test,"reg2_thes_lasso")
elasticnet(X_train, y_train, X_test, y_test,"reg2_thes_elasticnet")
dtreeregressor(X_train, y_train, X_test, y_test,"reg2_thes_dtreeregressor")
gradientboosting(X_train, y_train, X_test, y_test,"reg2_thes_gradientboostingreg")
randomforest(X_train, y_train, X_test, y_test,"reg2_thes_randomforestreg")