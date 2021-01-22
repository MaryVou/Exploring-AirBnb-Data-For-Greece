"""
This part contains two functions that will prepare the dataset in order to use it in regression and classification parts.
After running columns_picker.py to help choose the right columns manually, one has to insert those columns in this script's "columns" list.
The mergeDataframes function is only usefull if someone is working with both listing files, and the detailed one misses the "price" column.
Otherwise it's not needed and the dataframes can be merged using the pandas.concat function.
The preprocess function is using the original columns that were picked.    
"""

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################
#                      DATAFRAME BUILDING PART                    #
###################################################################

#This function is only needed if one is using both listing files and the detailed one misses the "price" column
#For more recent detailed listing files use prepareDataframe()

def mergeDataframes(columns, file1, file2, city, month, cityCenter):	#Merges files of same month and city but of different size

	df1 = pd.read_csv(file1) 		#df1 should be the "short" file which will be used to extract "price" column
	df2 = pd.read_csv(file2)		#df2 should be the "long" file which contains all the columns

	df2["price"] = df2["id"].map(df1.set_index("id")["price"]) 	#creates a column "price" in the long dataframe
																#where ids match

	df2["dist_from_center"] = df2.apply(lambda x:geodesic((x["latitude"], x["longitude"]) , cityCenter) , axis=1) #adds column "dist_from_center"
	if "dist_from_center" not in columns:
		columns.append("dist_from_center")

	df2 = df2[columns]	#subsets
	
	df2["city"] = city
	df2['month'] = month 	#returns a merged dataframe
	return df2

def prepareDataframe(columns,file,city,month,cityCenter):

	df = pd.read_csv(file)

	df["dist_from_center"] = df.apply(lambda x:geodesic((x["latitude"], x["longitude"]) , cityCenter) , axis=1) #adds column "dist_from_center"
	if "dist_from_center" not in columns:
		columns.append("dist_from_center")

	df = df[columns]	#subsets
	
	df["city"] = city
	df['month'] = month 	#returns a merged dataframe
	return df


"""
PREPROCESSING PART

COLUMNS WILL BE DIVIDED TO CONTINUOUS AND CATEGORICAL

CATEGORICAL --> one hot encoder

SPECIAL PREPROCESSING:
host_since --> keep only the year for simplicity and change type to int
dist_from_center --> get rid of "km" and change type to float
"""
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", 150)

###################################################################
#                      DATA PREPROCESSING PART                    #
###################################################################

def preprocess(pathToDf,continuous,categorical):

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
	#                    DEAL WITH CATEGORICAL VALUES				    #
	#####################################################################

	for column in categorical:
		df[column] = df[column].astype("category").cat.codes
 	
	for column in categorical:
		ohe = pd.get_dummies(df[column], prefix = column)
		df = pd.concat([df, ohe], axis = 1)
		df = df.drop([column], axis = 1)

	#####################################################################
	#                     		 REMOVE OUTLIERS	         	        #
	#####################################################################
	
	Q1 = df['price'].quantile(0.25)
	Q3 = df['price'].quantile(0.75)
	IQR = Q3 -Q1

	df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 +1.5 * IQR)))]
	
	#####################################################################
	#                         SAVE CHANGES					            #
	#####################################################################
	
	print("\nData Representation After Preprocessing:\n",df.head(10))
	print(df.shape)
	df.to_csv(pathToDf)