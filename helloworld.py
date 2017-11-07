"""
Hello world K means clustering
~~~~~~~~~~~
"""
print(__doc__)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv ('data_1024.csv')
print df

f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values


print ("This line will be printed.")
x = 1
if x == 1: print ("x is 1")
y = 0
if y < 1: print ("y is less then 1")