import pandas as pd
import regex as re


# Load the first CSV file
file_path1 = 'data/PMBJP-Product.csv'  
df1 = pd.read_csv(file_path1)

# Load the second CSV file
file_path2 = 'data/Pradhan Mantri Bhartiya Jan Aushadhi Pariyojna.csv'  
df2 = pd.read_csv(file_path2)


# Merge the two DataFrames on 'Generic Name of the Medicine'
merged_df = pd.merge(df1, df2, on='Drug Code', suffixes=('_1', '_2'), how='inner')


df = merged_df

df = df.drop(columns=['Sr.\nNo._2', 'Generic Name of the Medicine_2', 'Unit Size_2', 'Therapeutic Category_2'])
df.rename(columns={'MRP':'MRP_gvmt'}, inplace=True)


#print(df['MRP_market'] if df['MRP_market'].str.isalpha().all else 1)

df['difference'] = df['MRP_market'] - df['MRP_gvmt']

print(df.dtypes)

df.to_csv('data/medicine_data.csv', index=False)
