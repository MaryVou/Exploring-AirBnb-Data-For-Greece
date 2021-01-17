import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", 150)

###################################################################
#                      DATA PREPROCESSING PART                    #
###################################################################

#print correlations higher than 0.6
def locateStrongCorrelations(df):
	df1 = df
	df1 = df1.drop(['city','month'],axis=1)	
	print("#################STRONGEST CORRELATIONS#################")
	for i in df1.columns:
		for j in df1.columns:
			if ((df1[i].corr(df1[j]) >= 0.6) and (i != j)):
				print(i+"-"+j,df1[i].corr(df1[j]))
	print("########################################################")

def preprocess(pathToDf):

	df = pd.read_csv(pathToDf)

	del df["id"], df["latitude"], df["longitude"]	#not needed anymore

	print("\nData Representation Before Preprocessing:\n",df.head(10))
	print(df.shape)
	
	#####################################################################
	#                       DEAL WITH NA VALUES						    #
	#####################################################################	

	#Fill na rows with the most frequent value of the column

	#most common bed type by far is "Real Bed"
	df["bed_type"] = df["bed_type"].replace(np.nan,"Real Bed")

	#rows that don't have a value in "bathrooms" are probably just one 
	df["bathrooms"] = df["bathrooms"].replace(np.nan,1.0)
		
	#rows that don't have a value in "cleaning_fee" are probably free of charge so replace nan with zeros 
	df["cleaning_fee"] = df["cleaning_fee"].str.replace("$","")
	df["cleaning_fee"] = df["cleaning_fee"].replace(np.nan, 0)
	df["cleaning_fee"] = df["cleaning_fee"].astype(float)

	#rows that don't have a value in "security_deposit" are probably 0
	df["security_deposit"] = df["security_deposit"].str.replace("$","")
	df["security_deposit"] = df["security_deposit"].str.replace(",","")
	df["security_deposit"] = df["security_deposit"].replace(np.nan, 0)
	df["security_deposit"] = df["security_deposit"].astype(float)
	
	#rows that don't have a value in "extra_people" are probably 0
	df["extra_people"] = df["extra_people"].str.replace("$","")
	df["extra_people"] = df["extra_people"].replace(np.nan, 0)
	df["extra_people"] = df["extra_people"].astype(float)

	#drop the rest na values
	df.dropna(inplace=True)

	#####################################################################
	#                        CREATE NEW COLUMNS				      	    #
	#####################################################################

	#create 4 large categories for"dist_from_center" - close, relatively close, relatively far and far

	df["dist_from_center"] = df["dist_from_center"].astype(str).str[:-3].astype(float)
	df["dist_from_center"] = pd.cut(df["dist_from_center"],
		bins=[-1, 1, 3, 5, 15],
		labels=["very close", "relatively close", "relatively far", "very far"])

	#create new column "popularity" based on the number of reviews

	df["popularity"] = pd.cut(df["number_of_reviews"],
		bins=[-1, 10, 100, 500, 1000],
		labels=["undiscovered", "relatively popular", "popular", "sought after"])
	df = df.drop(["number_of_reviews"],axis=1)

	#create new column "availability" based on how many days the year, a property is available

	df["availability"] = pd.cut(df["availability_365"],
		bins=[-1, 179, 180, 364, 365],
		labels=["less than half year", "half year", "more than half year", "all year"])
	df = df.drop(["availability_365"],axis=1)

	#create new column "host_experience" based on how many properties one is managing 

	df["host_experience"] = pd.cut(df["calculated_host_listings_count"], 
		bins=[0, 1, 5, 10, 150], 
		labels=["home owner", "experienced", "very experienced","expert"])
	df = df.drop(["calculated_host_listings_count"], axis=1)

	#####################################################################
	#                            DATA SLICING						    #
	#####################################################################		

	df["host_since"] = df["host_since"].astype(str).str[:4].astype(int)

	#####################################################################
	#             SEPARATE CATEGORICAL AND CONTINUOUS DATA				#
	#####################################################################	

	#categorical values will be given codes and one hot encoding

	categorical = ["popularity","availability","bathrooms",
"bed_type","host_experience","dist_from_center", 
"host_is_superhost","host_identity_verified","property_type",
"room_type","accommodates", "guests_included","cancellation_policy",
"host_has_profile_pic","instant_bookable"]
	
	#continuous values will be scaled and have outliers removed

	continuous = ["price","cleaning_fee","extra_people",
	"host_since","security_deposit"]

	#only two columns will remain the same so that we can use them later: city and month

	#####################################################################
	#                    DEAL WITH CATEGORICAL VALUES				    #
	#####################################################################

	for column in categorical:
		df[column] = df[column].astype("category").cat.codes

	#now i can call locateStrongCorrelations()
	locateStrongCorrelations(df)

	for column in categorical:
		ohe = pd.get_dummies(df[column], prefix = column)
		df = pd.concat([df, ohe], axis = 1)
		df = df.drop([column], axis = 1)

	#####################################################################
	#                      REMOVE OUTLIERS FROM PRICE			        #
	#####################################################################
	
	Q1 = df["price"].quantile(0.25)
	Q3 = df["price"].quantile(0.75)
	IQR = Q3 -Q1

	df = df[~((df["price"] < (Q1 - 1.5 * IQR)) | (df["price"] > (Q3 +1.5 * IQR)))]
	
	#####################################################################
	#                      SCALE CONTINUOUS DATA					    #
	#####################################################################
	
	scaled_features = df.copy()
	features = scaled_features[continuous]
	scaler = StandardScaler().fit(features.values)
	features = scaler.transform(features.values)
	scaled_features[continuous] = features
	
	#####################################################################
	#                         SAVE CHANGES					            #
	#####################################################################

	
	print("\nData Representation After Preprocessing:\n",df.head(10))
	print(df.shape)
	df.to_csv(pathToDf)
	"""
	print("\nData Representation After Preprocessing:\n",scaled_features.head(10))
	print(scaled_features.shape)
	scaled_features.to_csv(pathToDf)
	"""
preprocess("final_df.csv")