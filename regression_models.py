from sklearn.metrics import r2_score,mean_absolute_error,max_error
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 0

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

################################################
#                 Functions 		           #
################################################

def printMetrics(y_test, predictions, model_name, plot_name):
	print("\n###############################################################")
	print('\n'+model_name)
	print('\nR2 Score: ',r2_score(y_test,predictions))
	print('Mean Absolute Error: ',mean_absolute_error(y_test,predictions))
	print('Max Error:',max_error(y_test,predictions))
	print("\n###############################################################")

	fig1, ax1 = plt.subplots()
	ax1.scatter(y_test, predictions)
	ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	fig1.suptitle(model_name,fontsize=12)
	plt.savefig("plots/"+plot_name+".png")

################################################
#               	Models 		               #
################################################

def linear(X_train, y_train, X_test, y_test,plot_name):
	lineReg = LinearRegression()
	lineReg.fit(X_train, y_train)
	predictions = lineReg.predict(X_test)

	printMetrics(y_test,predictions,"Linear",plot_name)

def ridge(X_train, y_train, X_test, y_test,plot_name):
	ridge = Ridge()
	ridge.fit(X_train, y_train)
	predictions = ridge.predict(X_test)

	printMetrics(y_test,predictions,"Ridge",plot_name)

def lasso(X_train, y_train, X_test, y_test,plot_name):
	lasso = LassoCV()
	lasso.fit(X_train, y_train)
	predictions = lasso.predict(X_test)
	
	printMetrics(y_test,predictions,"LassoCV",plot_name)

def elasticnet(X_train, y_train, X_test, y_test,plot_name):
	el = ElasticNetCV()
	el.fit(X_train, y_train)
	predictions = el.predict(X_test)

	printMetrics(y_test,predictions,"ElasticNet",plot_name)

def dtreeregressor(X_train, y_train, X_test, y_test,plot_name):
	dtree = DecisionTreeRegressor()
	dtree.fit(X_train, y_train)
	predictions = dtree.predict(X_test)
	
	printMetrics(y_test,predictions,"Decision Tree Regressor",plot_name)

def gradientboosting(X_train, y_train, X_test, y_test,plot_name):
	gb = GradientBoostingRegressor()
	gb.fit(X_train, y_train)
	predictions = gb.predict(X_test)

	printMetrics(y_test,predictions,"Gradient Boosting Regressor",plot_name)

def randomforest(X_train, y_train, X_test, y_test,plot_name):
	rf = RandomForestRegressor(n_estimators=100)
	rf.fit(X_train, y_train)
	predictions = rf.predict(X_test)

	printMetrics(y_test,predictions,"Random Forest Regressor",plot_name)