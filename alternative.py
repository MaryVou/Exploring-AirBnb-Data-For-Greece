"""
This script prepares the dataframe for the third part of the project,which refers to a more recent dataset.
The tricky part of this process is that the older and the newer datasets have many differences.
So we will apply some extra preprocessing here in order to be able to call the pandas.concat function 
After that the saved csv can be given as argument to run regression and classification scripts.  
"""

from dataframe_builder import preprocess, prepareDataframe, mergeDataframes
from regression_models import randomforest
from classification_models import rfc

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Let's suppose that we have already ran the columns_picker.py script and it helped us choose the below columns:
#NOTE: they are not the same columns with main.py
#Some of them weren't included in the most recent dataset
#REMOVED: 'extra_people', 'guests_included', 'cancellation_policy', 'bed_type', 'require_guest_phone_verification', 'require_guest_profile_picture'

columns = ['id', 'host_since', 'host_is_superhost', 'host_has_profile_pic', 
'host_identity_verified', 'latitude', 'longitude', 'property_type', 
'room_type', 'accommodates', 'minimum_minimum_nights',  
'minimum_maximum_nights', 'availability_365', 'number_of_reviews', 
'review_scores_value', 'instant_bookable', 
'calculated_host_listings_count','calculated_host_listings_count_private_rooms', 
'calculated_host_listings_count_shared_rooms', 'price']

"""OTHER COLUMNS THAT COULD BE USED:

->host_location to check if host is a local
->host_about could be used for text mining
->review_host_location + neighbourhood_cleansed to find popular neighbourhoods
"""

df1 = mergeDataframes(columns,"airbnb_data/ath_jul_short.csv","airbnb_data/ath_jul_long.csv","Athens","July",(37.9715, 23.7257))
df2 = mergeDataframes(columns,"airbnb_data/ath_jun_short.csv","airbnb_data/ath_jun_long.csv","Athens","June",(37.9715, 23.7257))
df3 = mergeDataframes(columns,"airbnb_data/thes_jul_short.csv","airbnb_data/thes_jul_long.csv","Thessaloniki","July",(40.6264, 22.9484))
df4 = mergeDataframes(columns,"airbnb_data/thes_jun_short.csv","airbnb_data/thes_jun_long.csv","Thessaloniki","June",(40.6264, 22.9484))

new_df = prepareDataframe(columns,"airbnb_data/crete_dec.csv","Crete","December",(35.5138,24.0180))

#Special changes because newer and older files contain some differences
new_df["price"] = new_df["price"].astype(str).str[1:]
new_df["price"] = new_df["price"].str.replace(",",".")
new_df["price"] = new_df["price"].str[:-3]
new_df["price"] = new_df["price"].astype(float)

#now we can call concat

df = pd.concat([df1,df2,df3,df4,new_df],axis=0)

del df["id"], df["latitude"], df["longitude"]	#not needed anymore

###################################################################
#                      DEALING WITH NA VALUES                     #
###################################################################

print("DATAFRAME SHAPE BEFORE NAN VALUES PREPROCESSING: ",df.shape)

#drop na values
df.dropna(inplace=True)

print("DATAFRAME SHAPE AFTER NAN VALUES PREPROCESSING: ",df.shape)

df.to_csv("alt_df.csv", index=False)

#####################################################################
#             SEPARATE CATEGORICAL AND CONTINUOUS DATA				#
#####################################################################	
	
#categorical values will be given codes and one hot encoding

categorical = ["host_is_superhost","host_has_profile_pic","host_identity_verified",
	"property_type","room_type","instant_bookable","dist_from_center","availability",
	"popularity","host_experience","calculated_host_listings_count_private_rooms", 
"calculated_host_listings_count_shared_rooms","host_years","review_scores_value"]
	
#continuous values will be scaled and have outliers removed

continuous = ["price","minimum_minimum_nights","minimum_maximum_nights","accommodates"]

#only two columns will remain the same so that we can use them later: city and month

###################################################################
#                 CALL PREPROCESSING FUNCTION                     #
###################################################################

preprocess("alt_df.csv",continuous,categorical)

###################################################################
#                 			TESTING AREA                          #
###################################################################

df = pd.read_csv("alt_df.csv")

if "Unnamed: 0" in df.columns:
	del df["Unnamed: 0"]

print("###################################################################")
print("                			TEST REGRESSION                           ")
print("###################################################################\n")

#FIRST TEST: train on athens and test on crete

df_ath = df.loc[df['city'] == 'Athens']
df_cre = df.loc[df['city'] == 'Crete']

X_train = np.array(df_ath.drop(['price','city','month'],axis=1))
y_train = np.array(df_ath['price'])

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1, shuffle=True)

p_test = int(len(X_train)*0.2)

X_test = np.array(df_cre.drop(['price','city','month'],axis=1))
y_test = np.array(df_cre['price'])

X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1, shuffle=True)

X_test = X_test[:p_test]
y_test = y_test[:p_test]

print("\nTRAINING ON ATHENS DATA AND TESTING ON CRETE")
randomforest(X_train, y_train, X_test, y_test,"gen_reg_randomforest_3")

#SECOND TEST: train on thessaloniki and test on crete

df_thes = df.loc[df['city'] == 'Thessaloniki']

X_train = np.array(df_thes.drop(['price','city','month'],axis=1))
y_train = np.array(df_thes['price'])

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1, shuffle=True)

p_test = int(len(X_train)*0.2)

X_test = np.array(df_cre.drop(['price','city','month'],axis=1))
y_test = np.array(df_cre['price'])

X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1, shuffle=True)

X_test = X_test[:p_test]
y_test = y_test[:p_test]

print("\nTRAINING ON THESSALONIKI DATA AND TESTING ON CRETE")
randomforest(X_train, y_train, X_test, y_test,"gen_reg_randomforest_4")

#THIRD TEST: train on both Athens and Thessaloniki and test on Crete

df_mixed = df.loc[(df['city'] == 'Athens') | (df['city'] == 'Thessaloniki')]

X_train = np.array(df_mixed.drop(['price','city','month'],axis=1))
y_train = np.array(df_mixed['price'])

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1, shuffle=True)

p_test = int(len(X_train)*0.2)

X_test = np.array(df_cre.drop(['price','city','month'],axis=1))
y_test = np.array(df_cre['price'])

X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1, shuffle=True)

X_test = X_test[:p_test]
y_test = y_test[:p_test]

print("\nTRAINING ON BOTH ATHENS AND THESSALONIKI DATA AND TESTING ON CRETE")
randomforest(X_train, y_train, X_test, y_test,"gen_reg_randomforest_5")

print("###################################################################")
print("                 	TEST CLASSIFICATION                           ")
print("###################################################################\n")

#FIRST TEST: train on athens and test on crete

df_ath = df.loc[df['city'] == 'Athens']
del df_ath['city']
del df_ath['month']

df_cre = df.loc[df['city'] == 'Crete']
del df_cre['city']
del df_cre['month']

df_ath['range'] =pd.cut(df_ath['price'], bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_ath = df_ath.drop(["price"],axis=1)

X_train = np.array(df_ath.drop(['range'],axis=1))
y_train = np.array(df_ath['range'])


df_cre['range'] = pd.cut(df_cre['price'],bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_cre = df_cre.drop(["price"],axis=1)

X_test = np.array(df_cre.drop(['range'],axis=1))
y_test = np.array(df_cre['range'])

print("\nTRAINING ON ATHENS DATA AND TESTING ON CRETE")
rfc(X_train, y_train, X_test, y_test,"")

#SECOND TEST: train on thessaloniki and test on crete

df_thes = df.loc[df['city'] == 'Thessaloniki']
del df_thes['city']
del df_thes['month']

df_thes['range'] =pd.cut(df_thes['price'], bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_thes = df_thes.drop(["price"],axis=1)

X_train = np.array(df_thes.drop(['range'],axis=1))
y_train = np.array(df_thes['range'])

print("\nTRAINING ON THESSALONIKI DATA AND TESTING ON CRETE")
rfc(X_train, y_train, X_test, y_test,"")

#THIRD TEST: train on both Athens and Thessaloniki and test on Crete

df_mixed = df.loc[(df['city'] == 'Thessaloniki') | (df['city'] == 'Athens')]
del df_mixed['city']
del df_mixed['month']

df_mixed['range'] =pd.cut(df_mixed['price'], bins=[0,50,100,200,500],labels=["Less than 50","50-100","100-200","More than 200"], include_lowest=True)

df_mixed = df_mixed.drop(["price"],axis=1)

X_train = np.array(df_mixed.drop(['range'],axis=1))
y_train = np.array(df_mixed['range'])

print("\nTRAINING ON BOTH ATHENS AND THESSALONIKI DATA AND TESTING ON CRETE")
rfc(X_train, y_train, X_test, y_test,"")