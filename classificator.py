from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

dataX  = pd.read_csv('./dataset/x_values.csv')
y = pd.read_csv('./dataset/y_values.csv') 

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(dataX, np.ravel(y))


