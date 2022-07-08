import pandas as pd
import numpy as np
import os

#load data set into df
filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#randomly select 30% of data examples
percentage = 0.3
num_rows = df.shape[0]
#df_subset = df.loc[np.random.choice(df.index, size=int(percentage*num_rows), replace=False)] numpy way
df_subset = df.sample(int(percentage*num_rows)) #pandas way

#filter by column values
condition = df['workclass'] == 'Private'
df_private = df[condition]
num_rows = df_private.shape[0]

#data analysis
condition = df['sex_selfID'] == 'Female'
df[condition]['age'].mean()

condition1 = (df['workclass'] == 'Local-gov')
condition2 = (df['hours-per-week'] > 40)
condition = condition1 & condition2
df_local = df[condition]
rows = df_local.shape[0]

#randomly sample 50% of rows where native country info is available
#(ignores missing values)
percentage = 0.5
df_country_notnull = df[df['native-country'].notnull()]
num_rows = df_country_notnull.shape[0]
df_filtered = df_country_notnull.loc[np.random.choice(df_country_notnull.index, size=int(percentage*num_rows), replace=False)]
mean_age = df_filtered['age'].mean()