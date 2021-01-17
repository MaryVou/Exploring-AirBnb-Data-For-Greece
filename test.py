import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

df = pd.read_csv("final_df.csv")

print(df.shape)
for column in df.columns:
	print(column+": ",df[column].count())

#guests_included
#cancellation_policy