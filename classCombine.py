# %%
# Combine individual participants' features into single large matrix, then export file

import numpy as np
import scipy as sp
import pandas as pd
import pickle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utilityFunctions import pairLoader

outNamData='GTP000_Data.csv'
outNamLabels='GTP000_Labels.csv'
X = np.genfromtxt(outNamData, delimiter=',')
#df = pd.read_csv(outNamData)
#X=df.to_numpy()
y = np.genfromtxt(outNamLabels, delimiter=',')

X = X.astype(float)
y = y.astype(int)
X=X[1:10510,0:560]

X[np.isnan(X)] = 0
X[np.isinf(X)] = 0
print(np.shape(X))

y[np.isnan(y)] = 0
y[np.isinf(y)] = 0
print(np.shape(y))
X=np.squeeze(X)
y=np.squeeze(y)

clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)

filename = 'gtExampleModel.sav'
pickle.dump(clf, open(filename, 'wb'))
#loaded_model = pickle.load(open(filename, 'rb'))