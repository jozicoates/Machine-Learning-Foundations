import pandas as pd
import numpy as np
import os

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#display summary statistics by column
df.describe(include='all')

#group columns into binary values
df['workclass'].unique()


#since there are only two values for self-employment, we simplify the code by writing
#NOT self employed

#create group 1: not-self-emp
#get all examples (rows) in which the workclass feature (columns) is not self-employed
columns_not_self_employed = ~(df['workclass'] == 'Self-emp-not-inc') & ~(df['workclass'] == 'Self-emp-inc')

#leave NaN (null) values in dataset for now; get all examples where workclass feature is not null
columns_not_null = ~(df['workclass'].isnull())
condition = columns_not_self_employed & columns_not_null

#change all of the workclass values that fulfill the specified condition to Not-self-emp
df['workclass'] = np.where(condition, 'Not-self-emp', df['workclass'])

#check to see new feature values for workclass
df['workclass'].unique()


#create group 2: self-emp
condition = (df['workclass'] == 'Self-emp-not-inc') | (df['workclass'] == 'Self-emp-inc')
df['workclass'] = np.where(condition, 'Self-emp', df['workclass'])

#check to see new feature values for workclass
df['workclass'].unique()


#transform label into binary 'True' if >50K income, else 'False'
condition1 = (df['label'] == '>50K')
df['label'] = np.where(condition1, 'True', df['label'])
condition2 = (df['label'] == '<=50K')
df['label'] = np.where(condition2, 'False', df['label'])


#CATEGORICAL VARIABLES
df['education'].dtype        #dtype('O') --> not ordered

#create correctly ordered list of category names:
edu = ['Preschool', '1st-4th', '5th-6th','7th-8th', '9th', '10th', '1th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
df['education'] = pd.Categorical(df['education'], ordered=True, categories=edu)

#check to make sure type changed
df['education'].dtype   #CategoricalDtype(categories=[...])


#convert categorical variables to "dummy" binary variables
df_binary = pd.get_dummies(df)