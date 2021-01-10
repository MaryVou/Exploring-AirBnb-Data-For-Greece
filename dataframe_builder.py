import pandas as pd
from geopy.distance import geodesic

columns = ["id","price","number_of_reviews",
"availability_365","latitude","longitude","cleaning_fee",
"bathrooms","bed_type","extra_people",
"calculated_host_listings_count",
"host_since","security_deposit","host_is_superhost","host_identity_verified",
"property_type","room_type","accommodates", "guests_included",
"cancellation_policy","host_has_profile_pic","instant_bookable"]

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

df.to_csv("final_df.csv", index=False)
