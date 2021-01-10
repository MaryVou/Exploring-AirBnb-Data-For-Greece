import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("final_df.csv")
del df['Unnamed: 0']

########################################
#           Experiment 1               #
########################################

#Train the dataset with "Athens" data (both months) and use "Thessaloniki" as test
df_ath = df[df['location'] == 0]
del df_ath['location']

df_thes = df[df['location'] == 1]
del df_thes['location']

X = np.asarray(df_ath[["host_is_superhost","room_type","accommodates","month","deviation_from_mean_price","dist_from_center"]])
Y = np.asarray(df_ath["price"])

#Test for Athens first
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle= True) 
lineReg = LinearRegression().fit(X_train, y_train)
print("Train: Athens\nTest: Athens\nScore: ",lineReg.score(X_test, y_test))

predictions = lineReg.predict(X_test)

print("\nActual Price vs Predicted Price")
for i in range(5):
	print(y_test[i],"\t\t",predictions[i])

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, predictions)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax1.set_xlabel('Measured')
ax1.set_ylabel('Predicted')
fig1.suptitle("Train Athens and Test on Athens",fontsize=12)
plt.savefig("plots/ath-ath.png")

print("\n############################################")

#Now test on Thessaloniki
sample = df_thes.sample(n=int(len(X_train)*0.1))
X_test = np.asarray(sample[["host_is_superhost","room_type","accommodates","month","deviation_from_mean_price","dist_from_center"]])
y_test = np.asarray(sample['price'])

print("\nTrain: Athens\nTest: Thessaloniki\nScore: ",lineReg.score(X_test, y_test))

predictions = lineReg.predict(X_test)

print("\nActual Price vs Predicted Price")
for i in range(5):
	print(y_test[i],"\t\t",predictions[i])

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, predictions)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax2.set_xlabel('Measured')
ax2.set_ylabel('Predicted')
fig2.suptitle("Train Athens and Test on Thessaloniki",fontsize=12)
plt.savefig("plots/ath-thes.png")

print("\n############################################")

del df_ath,df_thes,sample

########################################
#           Experiment 2               #
########################################

df_june = df[df['month'] == 0]
del df_june['month']

df_july = df[df['month'] == 1]
del df_july['month']

X = np.asarray(df_june[["host_is_superhost","room_type","accommodates","location","deviation_from_mean_price","dist_from_center"]])
Y = np.asarray(df_june["price"])

#Test for June first
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle= True) 
lineReg = LinearRegression().fit(X_train, y_train)
print("Train: June\nTest: June\nScore: ",lineReg.score(X_test, y_test))

predictions = lineReg.predict(X_test)

print("\nActual Price vs Predicted Price")
for i in range(5):
	print(y_test[i],"\t\t",predictions[i])

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, predictions)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax3.set_xlabel('Measured')
ax3.set_ylabel('Predicted')
fig3.suptitle("Train June and June on Athens",fontsize=12)
plt.savefig("plots/june-june.png")

print("\n############################################")

sample = df_july.sample(n=int(len(X_train)*0.1))
X_test = np.asarray(sample[["host_is_superhost","room_type","accommodates","location","deviation_from_mean_price","dist_from_center"]])
y_test = np.asarray(sample['price'])

print("\nTrain: June\nTest: July\nScore: ",lineReg.score(X_test, y_test))

predictions = lineReg.predict(X_test)

print("\nActual Price vs Predicted Price")
for i in range(5):
	print(y_test[i],"\t\t",predictions[i])

fig4, ax4 = plt.subplots()
ax4.scatter(y_test, predictions)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax4.set_xlabel('Measured')
ax4.set_ylabel('Predicted')
fig4.suptitle("Train June and Test on July",fontsize=12)
plt.savefig("plots/june-july.png")

del df_june,df_july,sample