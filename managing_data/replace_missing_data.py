import pandas as pd
import numpy as np
import os

filename = os.path.join(os.getcwd(), "data", "adult.data.partial.missing")
df = pd.read_csv(filename, header=0)

#identify missing values
df.isnull().values.any()

#number of occurances of missing values
nan_count = np.sum(df.isnull(), axis=0)

#store names of columns with missing values
condition = nan_count != 0
col_names = nan_count[condition].index
nan_cols = list(col_names)

#choose which values to fill-- cannot fill object/string entities with mean
nan_col_types = df[nan_cols].dtypes

#create 'dummy' variables for missing values
df['age_na'] = df['age'].isnull()

#fill in missing values
df.loc[df['age'].isnull()]

#compute mean for all non-null age values
mean_ages = df['age'].mean()

#fill all missing values with mean
df['age'].fillna(value=mean_ages, inplace=True)

#check to see if converted all missing values to mean value
np.sum(df['age'].isnull(), axis=0)