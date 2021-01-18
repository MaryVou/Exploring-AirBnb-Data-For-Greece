import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

columns = ['id', 'host_since', 'host_is_superhost', 'host_has_profile_pic', 
'host_identity_verified', 'latitude', 'longitude', 'property_type', 
'room_type', 'accommodates', 'bathrooms', 'bed_type', 'guests_included', 
'extra_people', 'minimum_minimum_nights',  
'minimum_maximum_nights', 'availability_365', 'number_of_reviews', 
'review_scores_value', 'instant_bookable', 
'calculated_host_listings_count', 'cancellation_policy', 
'require_guest_profile_picture', 'require_guest_phone_verification',  
'calculated_host_listings_count_private_rooms', 
'calculated_host_listings_count_shared_rooms', 'price']

#will use host_location to check if host is a local
#host_about could be used for text mining
#review_host_location + neighbourhood_cleansed to find popular neighbourhoods

###################################################################
#                      DATAFRAME BUILDING PART                    #
###################################################################

def mergeDataframes(file1, file2, city, month, cityCenter):	#Merges files of same month and city but of different size
	
	global columns

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

df1 = mergeDataframes("airbnb_data/ath_jul_short.csv","airbnb_data/ath_jul_long.csv","Athens","July",(37.9715, 23.7257))
df2 = mergeDataframes("airbnb_data/ath_jun_short.csv","airbnb_data/ath_jun_long.csv","Athens","June",(37.9715, 23.7257))
df3 = mergeDataframes("airbnb_data/thes_jul_short.csv","airbnb_data/thes_jul_long.csv","Thessaloniki","July",(40.6264, 22.9484))
df4 = mergeDataframes("airbnb_data/thes_jun_short.csv","airbnb_data/thes_jun_long.csv","Thessaloniki","June",(40.6264, 22.9484))

df = pd.concat([df1, df2, df3, df4], axis = 0)

del df["id"], df["latitude"], df["longitude"]	#not needed anymore

###################################################################
#                      DEALING WITH NA VALUES                     #
###################################################################

print("DATAFRAME SHAPE BEFORE NAN VALUES PREPROCESSING: ",df.shape)

#Fill na rows with the most frequent value of the column

#most common bed type by far is "Real Bed"
df["bed_type"] = df["bed_type"].replace(np.nan,"Real Bed")

#rows that don't have a value in "bathrooms" are probably just one 
df["bathrooms"] = df["bathrooms"].replace(np.nan,df["bathrooms"].mean())
	
#rows that don't have a value in "extra_people" are probably 0
df["extra_people"] = df["extra_people"].str.replace("$","").astype(float)
df["extra_people"] = df["extra_people"].replace(np.nan, df["extra_people"].mean())

df['guests_included'] = df['guests_included'].replace(np.nan,df["guests_included"].mean())

#drop the rest na values
df.dropna(inplace=True)

print("DATAFRAME SHAPE AFTER NAN VALUES PREPROCESSING: ",df.shape)

df.to_csv("final_df.csv", index=False)