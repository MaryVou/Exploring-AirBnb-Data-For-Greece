"""
PREPROCESSING

COLUMNS WILL BE DIVIDED TO CONTINUOUS AND CATEGORICAL

CATEGORICAL --> one hot encoder

SPECIAL PREPROCESSING:
host_since --> keep only the year for simplicity

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", 150)

###################################################################
#                      DATA PREPROCESSING PART                    #
###################################################################

def preprocess(pathToDf):

	df = pd.read_csv(pathToDf)

	print("\nData Representation Before Preprocessing:\n",df.head(10))
	print(df.shape)

	#####################################################################
	#                            DATA SLICING						    #
	#####################################################################		

	df["dist_from_center"] = df["dist_from_center"].astype(str).str[:-3].astype(float)
	df["host_since"] = df["host_since"].astype(str).str[:4].astype(int)

	#####################################################################
	#                        CREATE NEW COLUMNS				      	    #
	#####################################################################
	
	#create 4 large categories for"dist_from_center" - close, relatively close, relatively far and far

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

	#create new column "host_years" based on how many years one is a host
	
	df["host_years"] = 2020 - df["host_since"]
	df["host_years"] = pd.cut(df["host_years"], 
		bins=[-1, 1, 3, 10, 15], 
		labels=["one or less", "one to three", "three to ten","over ten"])
	df = df.drop(["host_since"], axis=1)


	#####################################################################
	#             SEPARATE CATEGORICAL AND CONTINUOUS DATA				#
	#####################################################################	
	
	#categorical values will be given codes and one hot encoding

	categorical = ["host_is_superhost","host_has_profile_pic","host_identity_verified",
	"property_type","room_type","bathrooms","bed_type","guests_included",
	"instant_bookable","cancellation_policy","require_guest_profile_picture", 
	"require_guest_phone_verification","dist_from_center","availability","popularity",
	"host_experience","calculated_host_listings_count_private_rooms", 
"calculated_host_listings_count_shared_rooms","host_years","review_scores_value"]
	
	#continuous values will be scaled and have outliers removed

	continuous = ["price","minimum_minimum_nights","minimum_maximum_nights",
	"extra_people","accommodates"]

	#only two columns will remain the same so that we can use them later: city and month

	#####################################################################
	#                    DEAL WITH CATEGORICAL VALUES				    #
	#####################################################################

	for column in categorical:
		df[column] = df[column].astype("category").cat.codes

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
	
	#tried scaling continuous features but didn't get good results

	"""
	scaled_features = df.copy()
	features = scaled_features[continuous]
	scaler = StandardScaler().fit(features.values)
	features = scaler.transform(features.values)
	scaled_features[continuous] = features
	"""
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