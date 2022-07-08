import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme() #makes plots look better

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#make histogram of age column
sns.histplot(data=df, x='age')

#historgram of logarithm (of age)
sns.histplot(data=df, x='age', log_scale=True)

#rescale plot to zoom 
plt.ylim(0,600)

#bar plot for categorical feature
sns.histplot(data=df, x='education')

#increase readability in bar plot
fig1 = plt.figure(figsize=(13,7)) #resize image
ax = sns.histplot(data=df, x='education')
t1 = plt.xticks(rotation=45) #rotate x-axis labels

#enforce order of categorical variables
cat_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
df['education'] = pd.Categorical(df['education'], cat_order)

fig2 = plt.figure(figsize=(13,7)) 
ax = sns.histplot(data=df, x="education")
t2 = plt.xticks(rotation=45)