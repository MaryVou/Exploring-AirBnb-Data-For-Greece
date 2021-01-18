import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

non_useful_columns = ['listing_url', 'scrape_id', 'last_scraped', 'name', 
'description', 'neighborhood_overview', 'picture_url','host_id', 'host_url', 
'host_name','host_location', 'host_about', 'host_response_time', 
'host_response_rate', 'host_acceptance_rate','host_thumbnail_url', 
'host_picture_url', 'host_neighbourhood','host_listings_count','host_total_listings_count', 
'host_verifications','neighbourhood', 'neighbourhood_cleansed', 
'neighbourhood_group_cleansed','summary','space','notes','transit','access',
'interaction','house_rules','thumbnail_url','medium_url','xl_picture_url',
'street','city','state','zipcode','market','smart_location','country_code',
'country','is_location_exact','weekly_price','monthly_price','calendar_updated',
'calendar_last_scraped','first_review','last_review','license']

columns = []

df1 = pd.read_csv("airbnb_data/ath_jul_short.csv")
df2 = pd.read_csv("airbnb_data/ath_jul_long.csv")

df2["price"] = df2["id"].map(df1.set_index("id")["price"])

print(len(df2))

#First columns to be inserted in the dataframe will be the ones of interest and with little nan values

for col in df2.columns:
	na_values_percentage = df2[col].isna().sum()/len(df2)
	if na_values_percentage<=0.25 and col not in non_useful_columns:
		columns.append(col)

df2 = df2[columns]


print("\nCOLUMNS OF INTEREST WITH LITTLE NAN VALUES:")
print(columns)

#Secondly we will explore the contents of each chosen column

for col in df2.columns:
	print(col)
	print(df2[col].value_counts())
	print()

"""
COLUMNS REMOVED DUE TO NON USEFUL INFO:

experiences_offered (only none values), 
has_availability (always true), 
requires_license (always true), 
is_business_travel_ready (always false)
"""

#Finally we will remove highly correlated features

plt.figure(figsize=(16,6))
plt.savefig("plots/correlation.png")
heatmap = sns.heatmap(df2.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig("plots/correlation.png")

"""
By examining the graph we found some strongly correlated features and we removed them
"""