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


