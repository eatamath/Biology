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
        return []
    res = []
    l = len(X)
    if select==[]:
        for i in range(l):
            if dims==-1 and dime==-1:
                line = [0 for i in range(feature_num+1)]
                for k,v in X[i].items():
                    line[k] += v
                line[-1] = Y[i]
            else:
                line = [0 for i in range(dime-dims+2)]
                for k,v in X[i].items():
                    if k>=dime and k<=dims:
                        line[k-dims] = v
                line[-1] = Y[i]
            res.append(line)
            if i%batch==batch-1:
                print('batch',i/batch)
    else:
        for i in range(l):
            line = [0 for i in range(feature_num)]
            for k,v in X[i].items():
                line[k] = v
            line = np.array(line)[select].tolist()
            res.append(line)
    res = np.array(res,copy=False)
    return res
        
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