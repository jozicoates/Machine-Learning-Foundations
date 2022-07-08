import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#filter the dataset
df_sub = df[['age', 'capital-gain', 'hours-per-week', 'education', 'labe;']].copy()

#make a (clean) pairplot of numeric features (above)
sns.pairplot(data=df_sub, hue='label', plot_kws={'s':3})

#bar plot on categorical feature (with y=label)
cat_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
df_sub['education'] = pd.Categorical(df_sub['education'], cat_order)

#plot histogram of all levels of education on x-axis, counts on y-axis
#but, split every bar into two parts, depending on income
fig1 = plt.figure(figsize=(13,7))
t1 = plt.xticks(rotation=45)
sns.histplot(data=df_sub, x='education', hue='label', multiple='stack')