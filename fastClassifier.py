#running intra-subject tests

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate, train_test_split
from sklearn import metrics, linear_model, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, make_scorer, classification_report
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from scipy.stats import stats
from utilityFunctions import pairLoader, eegFeatureReducer, balancedMatrix, featureSelect, speedClass, dirClass, dualClass, fsClass, fastClassOutputs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm

# for N-fold cross validation
# set parameters
N=4
featureNumber=int(3)

allSubNames=list(['GTP000','GTP003','GTP004','GTP007','GTP044','GTP045','GTP047','GTP104','GTP223','GTP303','GTP308','GTP455','GTP545','GTP556','GTP762','GTP765','allSub'])
stringName='gtFastClassifierResults.csv'

# initialize lists
# qda/lda
ca1AList=list()
ca1FList=list()
cb1AList=list()
cb1FList=list()
ca1AsList=list()
cb1AsList=list()
# N Bayes
ca2AList=list()
ca2FList=list()
cb2AList=list()
cb2FList=list()
ca2AsList=list()
cb2AsList=list()
# svm
ca3AList=list()
ca3FList=list()
cb3AList=list()
cb3FList=list()
ca3AsList=list()
cb3AsList=list()

# knn
ca4AList=list()
ca4FList=list()
cb4AList=list()
cb4FList=list()
ca4AsList=list()
cb4AsList=list()

# load data
subName='GTP000'

for subName in allSubNames:
	[X,y]=pairLoader(subName)

# classify data
	ca1finalAcc,ca1finalF1,ca1As,ca3finalAcc,ca3finalF1,ca3As,ca4finalAcc,ca4finalF1,ca4As=fastClassOutputs(N,X,y,featureNumber)
# qda list 
	ca1AList.append(ca1finalAcc)
	ca1FList.append(ca1finalF1)

	ca1AsList.append(ca1As)


# nbayes list 


# svm list 
	ca3AList.append(ca3finalAcc)
	ca3FList.append(ca3finalF1)

	ca3AsList.append(ca3As)


# knn list 
	ca4AList.append(ca4finalAcc)
	ca4FList.append(ca4finalF1)

	ca4AsList.append(ca4As)


# combining all
zipped = list(zip(allSubNames, ca1AList, ca1FList, ca1AsList, ca3AList, ca3FList, ca3AsList, ca4AList, ca4FList, ca4AsList))
df = pd.DataFrame(zipped, columns=['SubName', 'QDA_Acc','QDA_F1', 'QDA_AUC', 'SVM_Acc','SVM_F1', 'SVM_AUC', 'KNN_Acc', 'KNN_F1', 'KNN_AUC'])
df.to_csv(stringName) 
