"""
This script prepares the dataframe for the main part of the project, meaning the data from Athens and Thessaloniki.
After that the saved csv can be given as argument to run regression and classification scripts.  
"""

from dataframe_builder import mergeDataframes, preprocess
import pandas as pd
import numpy as np

#Let's suppose that we have already ran the columns_picker.py script and it helped us choose the below columns:

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

df1 = mergeDataframes(columns,"airbnb_data/ath_jul_short.csv","airbnb_data/ath_jul_long.csv","Athens","July",(37.9715, 23.7257))
df2 = mergeDataframes(columns,"airbnb_data/ath_jun_short.csv","airbnb_data/ath_jun_long.csv","Athens","June",(37.9715, 23.7257))
df3 = mergeDataframes(columns,"airbnb_data/thes_jul_short.csv","airbnb_data/thes_jul_long.csv","Thessaloniki","July",(40.6264, 22.9484))
df4 = mergeDataframes(columns,"airbnb_data/thes_jun_short.csv","airbnb_data/thes_jun_long.csv","Thessaloniki","June",(40.6264, 22.9484))

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

df.to_csv("main_df.csv", index=False)

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

###################################################################
#                 CALL PREPROCESSING FUNCTION                     #
###################################################################

preprocess("main_df.csv",continuous,categorical)