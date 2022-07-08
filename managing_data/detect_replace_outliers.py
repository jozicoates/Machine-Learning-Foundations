import pandas as pd
import numpy as np
import os
import scipy.stats as stats

filename = os.path.join(os.getcwd(), "data", "adult.data.partial")
df = pd.read_csv(filename, header=0)

#compute percentile of columns
edu_90 = np.percentile(df['education'], 90)

#add column with windsorized version of original columns
df['education-num-win'] = stats.mstats.winsorize(df['education-num'], limits=[0.01, 0.01])

#Z-SCORES (numpy)
F = [4,6,3,-3,4,5,6,7,3,8,1,9,1,2,2,35,4,1]

F_std = np.std(F)
F_mean = np.mean(F)
zscores = [(value-F_mean) / F_std for value in F]

#Z-SCORES (scipy)
zscores = stats.zscore(df['hours-per-week'])

#z-scores for all values of numeric columns
df_zscores = df.select_dtypes(include=['number']).apply(stats.zscore)