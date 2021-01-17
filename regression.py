import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,max_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


################################################
#               	Models 		               #
################################################

def linear(X_train, y_train, X_test, y_test,plot_name):
	lineReg = LinearRegression()
	lineReg.fit(X_train, y_train)
	predictions = lineReg.predict(X_test)
	print("\n###############################################################")
	print('\nLinear')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("Linear",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

def ridge(X_train, y_train, X_test, y_test,plot_name):
	ridge = Ridge()
	ridge.fit(X_train, y_train)
	predictions = ridge.predict(X_test)
	print("\n###############################################################")
	print('\nRidge')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("Ridge",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

def lasso(X_train, y_train, X_test, y_test,plot_name):
	lasso = Lasso()
	lasso.fit(X_train, y_train)
	predictions = lasso.predict(X_test)
	print("\n###############################################################")
	print('\nLasso')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("Lasso",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

def elasticnet(X_train, y_train, X_test, y_test,plot_name):
	el = ElasticNet()
	el.fit(X_train, y_train)
	predictions = el.predict(X_test)
	print("\n###############################################################")
	print('\nElasticNet')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("ElasticNet",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

def dtreeregressor(X_train, y_train, X_test, y_test,plot_name):
	dtree = DecisionTreeRegressor()
	dtree.fit(X_train, y_train)
	predictions = dtree.predict(X_test)
	print("\n###############################################################")
	print('\nDecision Tree Regressor')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("Decision Tree Regressor",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

def gradientboosting(X_train, y_train, X_test, y_test,plot_name):
	gb = GradientBoostingRegressor()
	gb.fit(X_train, y_train)
	predictions = gb.predict(X_test)
	print("\n###############################################################")
	print('\nGradient Boosting Regressor')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("Gradient Boosting Regressor",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

def randomforest(X_train, y_train, X_test, y_test,plot_name):
	rf = RandomForestRegressor(n_estimators=100)
	rf.fit(X_train, y_train)
	predictions = rf.predict(X_test)
	print("\n###############################################################")
	print('\nRandom Forest Regressor')
	print('R2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle("Random Forest Regressor",fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

##########################################################
# 	PART A: TEST DIFFERENT MODELS ON THE WHOLE DATASET   #
##########################################################

df = pd.read_csv("final_df.csv")

del df["Unnamed: 0"]

X = np.array(df.drop(['price','city','month'],axis=1))
Y = np.array(df["price"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

linear(X_train, y_train, X_test, y_test,"test1_linear")
ridge(X_train, y_train, X_test, y_test,"test1_ridge")
lasso(X_train, y_train, X_test, y_test,"test1_lasso")
elasticnet(X_train, y_train, X_test, y_test,"test1_elasticnet")
dtreeregressor(X_train, y_train, X_test, y_test,"test1_dtreeregressor")
gradientboosting(X_train, y_train, X_test, y_test,"test1_gradientboostingreg")
randomforest(X_train, y_train, X_test, y_test,"test1_randomforestreg")

##########################################################
# 	PART B: TEST DIFFERENT MODELS ON DIFFERENT CITIES    #
##########################################################

#df1 = df.loc[:, ~df.columns.str.startswith("city")]
#df1 = df.loc[:, ~df.columns.str.startswith("month")]
