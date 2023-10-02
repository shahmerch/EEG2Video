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
from utilityFunctions import pairLoader, eegFeatureReducer, balancedMatrix, featureSelect, speedClass, dirClass, dualClass, fsClass, classOutputs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm

# for N-fold cross validation
# set parameters
N=4
featureNumber=int(3)

allSubNames=list(['GTP000','GTP003','GTP004','GTP007','GTP044','GTP045','GTP047','GTP104','GTP223','GTP303','GTP308','GTP455','GTP545','GTP556','GTP762','GTP765','allSub'])
stringName='gtClassifierResults.csv'

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
	ca1finalAcc,ca1finalF1,cb1fsAcc,cb1fsF1,ca2finalAcc,ca2finalF1,cb2fsAcc,cb2fsF1,ca3finalAcc,ca3finalF1,cb3fsAcc,cb3fsF1,ca4finalAcc,ca4finalF1,cb4fsAcc,cb4fsF1,ca1As,ca2As,ca3As,ca4As,cb1As,cb2As,cb3As,cb4As=classOutputs(N,X,y,featureNumber)
# qda list 
	ca1AList.append(ca1finalAcc)
	ca1FList.append(ca1finalF1)
	cb1AList.append(cb1fsAcc)
	cb1FList.append(cb1fsF1)
	ca1AsList.append(ca1As)
	cb1AsList.append(cb1As)

# nbayes list 
	ca2AList.append(ca2finalAcc)
	ca2FList.append(ca2finalF1)
	cb2AList.append(cb2fsAcc)
	cb2FList.append(cb2fsF1)
	ca2AsList.append(ca2As)
	cb2AsList.append(cb2As)

# svm list 
	ca3AList.append(ca3finalAcc)
	ca3FList.append(ca3finalF1)
	cb3AList.append(cb3fsAcc)
	cb3FList.append(cb3fsF1)
	ca3AsList.append(ca3As)
	cb3AsList.append(cb3As)

# knn list 
	ca4AList.append(ca4finalAcc)
	ca4FList.append(ca4finalF1)
	cb4AList.append(cb4fsAcc)
	cb4FList.append(cb4fsF1)
	ca4AsList.append(ca4As)
	cb4AsList.append(cb4As)

# combining all
zipped = list(zip(allSubNames, ca1AList, ca1FList, cb1AList, cb1FList, ca2AList, ca2FList, cb2AList, cb2FList, ca3AList, ca3FList, cb3AList, cb3FList, ca4AList, ca4FList, cb4AList, cb4FList,ca1AsList,cb1AsList,ca2AsList,cb2AsList,ca3AsList,cb3AsList,ca4AsList,cb4AsList))
df = pd.DataFrame(zipped, columns=['SubName', 'QDA_Acc','QDA_F1', 'FS_QDA_Acc','FS_QDA_F1', 'NB_Acc','NB_F1', 'FS_NB_Acc','FS_NB_F1', 'SVM_Acc','SVM_F1', 'FS_SVM_Acc','FS_SVM_F1','KNN_Acc','KNN_F1', 'FS_KNN_Acc','FS_KNN_F1','QDA_AUC','FS_QDA_AUC','NB_AUC','FS_NB_AUC','SVM_AUC','FS_SVM_AUC','KNN_AUC','FS_KNN_AUC'])
df.to_csv(stringName) 
