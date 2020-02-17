import numpy as np
import pandas as pd
import sklearn
import gc
import os
import re
import math
import sys
import random
from itertools import islice
import time
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import bokeh
import scipy as sp
from scipy import stats
from sklearn.feature_selection import *
import time
from joblib import dump, load
import configparser
import json

from sklearn.decomposition import *
from sklearn.ensemble import *
from collections import Counter
from sklearn.model_selection import *

import lightgbm as lgb
import sklearn.tree as Tr 

from module.module import *
from module.process import *


seed = 42
DATASET = 0
CONFIGURE_PATH = './conf.ini'

def GetConfigure():
    global PROTEIN_K,RNA_K,ENTROPY_IM,topK,RF_ENSENBLE,TRAIN_TEST_SPLIT
    global cfile,conf,commons
    global PARAMS
    cfile = configparser.ConfigParser()
    cfile.read(CONFIGURE_PATH)
    conf = dict(cfile.items(str(DATASET)))
    commons = dict(cfile.items('common'))

    PROTEIN_K = int( conf[str.lower('PROTEIN_K')] )
    RNA_K = int( conf[str.lower('RNA_K')] )
    ENTROPY_IM = float( conf[str.lower('ENTROPY_IM')] )
    topK = int( conf[str.lower('topK')] )
    RF_ENSENBLE = int( conf[str.lower('RF_ENSENBLE')] )
    TRAIN_TEST_SPLIT = float( commons[str.lower('TRAIN_TEST_SPLIT')] )
    PARAMS = conf['params']
    PARAMS = dict(eval(PARAMS))
    return

GetConfigure()



#### 读取数据

def ReadData():
    T=trainer(5,-1)
    data = T.MAIN_SINGLE_TEST(DATASET,PROTEIN_K,RNA_K)
    return data,T

# pair 文件输出

# OutputPairToFile(T.pairs,'./pairs'
#                  +str(DATASET)
#                  +str(PROTEIN_K)
#                  +str(RNA_K)
#                  +'.txt')

def ToMatrix(data,matrix_type='sparse'):
    res = Dict2Sparse(data)
    print('data shape %d %d'%(res[3],res[4]))
    arr = csr_matrix( ( np.array(res[2]),
                 (np.array(res[0]), np.array(res[1])) 
                ), shape = (res[3],res[4]) )
    if matrix_type=='dense':
        arr = arr.todense() ### matrix
        [X,Y] = [np.array(arr),np.array(data[0][1])]
        return [X,Y]
    return arr

#### 特征选择

def MutualInformationFeatureSelection(arr,data):
    mi = mutual_info_classif(arr,data[0][1],copy=False,n_neighbors=4)
    select = mi>ENTROPY_IM
    select = np.hstack([select,[False]])
    X = DictionaryToMatrix(data[0][0],data[0][1],feature_num=data[1]+1,select=select)
    Y = np.array(data[0][1])
    print('dimension ratio %f dimension remained %d'
          %(X.shape[1]/(data[1]+1),
            X.shape[1]))
    INFO('mutual information sum %f select %f'%(sum(mi),sum(mi[select[:-1]])/sum(mi)) )
    return [X,Y]

#### 数据集分割

def SplitDataset(X,Y):
    if TRAIN_TEST_SPLIT>0:
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X,Y,test_size=TRAIN_TEST_SPLIT,random_state = seed)
        del X,Y
        return [X_train,X_test,Y_train,Y_test]
    else:
        return None

#### 二次降维

def RandomForestDimensionalityReduction(X_train,X_test,Y_train,Y_test):
    rfclf = RandomForestClassifier(n_estimators=RF_ENSENBLE,n_jobs=3)
    rfclf.fit(X_train,Y_train)
    rf_fit_score = rfclf.score(X_train,Y_train)
    print('rf raw data fit score %f'%rf_fit_score)

    rf_select = list(zip([i for i in range(len(rfclf.feature_importances_))],rfclf.feature_importances_))
    rf_select = sorted(rf_select,key=lambda x:x[1],reverse=True)
    rf_select = list(map(lambda x:[x[0],x[1]],rf_select))
    rf_select = np.array(rf_select)

    print('select top K features importances',sum(rf_select[:topK,1]))
    rf_select = rf_select[:topK,0]
    rf_select = rf_select.astype('int32')
    X_train = X_train[:,rf_select]
    X_test = X_test[:,rf_select]
    print('dimension remained %d'%X_train.shape[1])
    return [X_train,X_test,Y_train,Y_test]

#### 预训练

def DecisionTreePrefit(X_train,X_test,Y_train,Y_test):
    dtc = Tr.DecisionTreeClassifier()
    dtc.fit(X_train,Y_train)
    print('pre fit score',dtc.score(X_train,Y_train))
    print('pre test score',dtc.score(X_test,Y_test))
    return

def Pipeline():
    T=trainer(5,-1)
    data = T.MAIN_SINGLE_TEST(DATASET,PROTEIN_K,RNA_K)
    OutputPairToFile(T.pairs,'./pairs'
                     +str(DATASET)
                     +str(PROTEIN_K)
                     +str(RNA_K)
                     +'.txt')
    arr = ToMatrix(data,'sparse')
    [X,Y] = MutualInformationFeatureSelection(arr,data)
    [X_train,X_test,Y_train,Y_test] = SplitDataset(X,Y)
    [X_train,X_test,Y_train,Y_test] = \
        RandomForestDimensionalityReduction(X_train,X_test,Y_train,Y_test)
    DecisionTreePrefit(X_train,X_test,Y_train,Y_test)
    LGBFit(X_train,X_test,Y_train,Y_test)
    return

def CleanVariables():
    global X,Y
    del X,Y
    return