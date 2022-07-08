import pandas as pd
import numpy as np
import os

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#get summary statistics, save in new df's
df_summ = df.describe()
df_summ_all = df.describe(include='all') #includes non-numerical data (NaN)

#get summary table for relevant data
describe_vars = ['age', 'education-num', 'hours-per-week']
df_summ_selected = df[describe_vars].describe()

#get percentile
age_25p = df_summ.loc['25%']['age']
print(f"The 25th percentile of the feature 'age' is {age_25p}")

#most variation
df_summ.loc['std'].idmax(axis=1)
#or, alternatively df_summ.idxmax(axis=1)['std']

#highest mean
column_name = df_summ.idxmax(axis=1)['mean']

#any negative values?
np.any(df_summ.loc['min'] < 0)

#highest range?
column_ranges = df_summ.loc['max'] - df_summ.loc['min']
column_range_name = column_ranges.idxmax(axis=1)