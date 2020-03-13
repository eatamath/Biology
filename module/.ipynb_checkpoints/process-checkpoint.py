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
import pymongo


client = pymongo.MongoClient('localhost')
db = client['reserch']

def DictionaryToMatrix(X,Y,feature_num,dims=-1,dime=-1,select=[],batch=10000):
    if len(X)!=len(Y):
        print('row number not match')
        return None
    res = []
    for row in X:
        if dims==-1 and dime==-1:
            xrow = [0 for i in range(feature_num)]
            for k,v in row.items():
                xrow[k] = v
            res.append(xrow)
        elif dims>=0 and dims<feature_num-1 and dime>0 and dime<feature_num:
            xrow = [0 for i in range(dims,dime)]
            for k,v in row.items():
                if k>=dims and k<dime:
                    xrow[k-dims] = v
            res.append(xrow)
    res = np.array(res)
    if select!=[]:
        if len(select)==dime-dims:
            res = res[:,select]
    return [res,np.array(Y)]
        
def OutputPairToFile(pairs,path):
    with open(path,'w+') as f:
        text = ''
        for k,v in pairs.items():
            k = k.split(',')
            text += k[0] +'\t'+ k[1] +'\t'+ str(v) +'\n'
        f.write(text)
    return

'''
    split symbol '@\n'
    dtype {protein, rna}
'''
def ExportSequence(sequence,dtype='protein'):
    try:
        if dtype=='protein':
            db['protein'].insert([sequence])
        elif dtype=='rna':
            db['rna'].insert([sequence])
    except Exception as e:
        print(e.args)
    return

'''
    split symbol '@\n'
    dtype {protein, rna}
'''
def ImportSequence(sequence,dtype='protein'):
    try:
        collections = ''
        if dtype=='protein':
            collections = 'protein'
        elif dtype=='rna':
            collections = 'rna'
        res = db[collections].find()[0]
    except Exception as e:
        print(e.args)
    return

'''
    split symbol '@'
'''
def WriteResult(DATASET,cv_result,conf,commons):
    with open('./results.txt','w+') as f:
        text = '@'+str(DATASET)+'\n'
        for k,v in commons.items():
            text += str(k)+"="+str(v)+'\n'
        for k,v in conf.items():
            text += str(k)+"="+str(v)+'\n'  
        for k,v in cv_result.items():
            text += str(k)+"="+str(np.max(v))+'\n'
        text += '\n'
        f.write(text)
    return

'''
    split symbol '@'
'''
def WriteDictResult(DATASET,dict_obj,fname):
    with open('./'+fname+'.txt','a+') as f:
        text = '@'+str(DATASET)+'\n'
        for k,v in dict_obj.items():
            text += str(k)+"=\t"+str(v)+'\n'
        text += '\n'
        f.write(text)
    return