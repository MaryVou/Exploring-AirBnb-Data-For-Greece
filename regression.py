import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from statistics import mean

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


################################################
#               Cross Validation               #
################################################

def crossValidation(X_train, y_train):

	lineReg = LinearRegression()
	cv_results = cross_validate(lineReg, X_train, y_train, cv=5, scoring = "r2")
	print("\nLinear: ",mean(cv_results['test_score']))

	ridgeReg = Ridge()
	cv_results = cross_validate(ridgeReg, X_train, y_train, cv=5, scoring = "r2")
	print("Ridge: ",mean(cv_results['test_score']))

	lassoReg = Lasso()
	cv_results = cross_validate(lassoReg, X_train, y_train, cv=5, scoring = "r2")
	print("Lasso: ",mean(cv_results['test_score']))

	elasticNetReg = ElasticNet()
	cv_results = cross_validate(elasticNetReg, X_train, y_train, cv=5, scoring = "r2")
	print("ElasticNet: ",mean(cv_results['test_score']))

	DTReg = DecisionTreeRegressor()
	cv_results = cross_validate(DTReg, X_train, y_train, cv=5, scoring = "r2")
	print("Desicion Tree Regressor: ",mean(cv_results['test_score']))

	KNReg = KNeighborsRegressor()
	cv_results = cross_validate(KNReg, X_train, y_train, cv=5, scoring = "r2")
	print("KNeighbors Regressor: ",mean(cv_results['test_score']))
	
	GBReg = GradientBoostingRegressor()
	cv_results = cross_validate(GBReg, X_train, y_train, cv=5, scoring = "r2")
	print("Gradient Boosting Regressor: ",mean(cv_results['test_score']))

	RFReg = RandomForestRegressor()
	cv_results = cross_validate(RFReg, X_train, y_train, cv=5, scoring = "r2")
	print("Random Forest Regressor: ",mean(cv_results['test_score']))

##########################################################
# 	PART A: TEST DIFFERENT MODELS ON THE WHOLE DATASET   #
##########################################################

df = pd.read_csv("final_df.csv")

#df1 = df.loc[:, ~df.columns.str.startswith("city")]
#df1 = df.loc[:, ~df.columns.str.startswith("month")]

featureColumns = df.drop(["price"],axis=1)

X = np.array(featureColumns)
Y = np.array(df["price"])

print("\nR2 SCORES OF DIFFERENT MODELS ON THE WHOLE DATASET")
crossValidation(X,Y)

##########################################################
# 	PART B: TEST DIFFERENT MODELS ON DIFFERENT DATASETS  #
##########################################################
