import pandas as pd
import numpy as np
import os

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#random sampling from data
percentage = 0.3
num_rows = df.shape[0]
df_subset = df.loc[np.random.choice(num_rows, int(num_rows*percentage))]

#verifying imbalance, check proportion female
unique_ssID = df['sex_selfID'].unique()
percent_female = np.sum(df_subset['sex_selfID'] == 'Female') / df_subset['sex_selfID'].shape[0]
counts = df_subset['sex_selfID'].value_counts()
counts['Female'] / sum(counts.values)

df_subset.groupby(['sex_selfID', 'label']).size()

#addressing imbalance- upsampling the underrepresented group
low_income_nonfemale, high_income_nonfemale = df_subset.groupby(['sex_selfID', 'label']).size()['Non-Female']
class_balance_nonfemale = high_income_nonfemale / low_income_nonfemale

low_income_female, high_income_female = df_subset.groupby(['sex_selfID', 'label']).size()['Female']
add_sample_size = int(class_balance_nonfemale * low_income_female - high_income_female) #we need this many more points in (Female) & (>50k) for balance

#subset the original data, exclude entires already in sample
df_never_sampled = df.drop(labels=df_subset.index, axis=0, inplace=False)

#filter the subset to include only the type of examples we want to upsample (females, higher income)
condition = (df_never_sampled['label'] == '>50K') & (df_never_sampled['sex_selfID'] == 'Female')
df_never_sampled_target = df_never_sampled[condition]

#sample from resulting set
size = min(add_sample_size, df_never_sampled_target.shape[0])

#append selected examples to original sample
rows = df.loc[np.random.choice(df_never_sampled_target.index, size=size, replace=False)]
df_balanced_subset = df_subset.append(rows)

#checks balance of new df
df_balanced_subset.groupby(['sex_selfID', 'label']).size()