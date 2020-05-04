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

#### data path configuration ####
class dataPathLoader:
    data_cluster_root = 'Negative1'
    fileroot = r'D:\StatisticsData\BIODATA\Data'
    hasNegative = [0,0,1,0,1,0,0]
    
    def __init__(self,datasetid):
        data_id=datasetid
        self.dataset_suit=['NPInter10412','reRPI2825','RPI488',
                           'RPI2241','RPI1807','LPI43250','EVLncRNAs']
        ### control dataset suit
        self.datasetname=self.dataset_suit[datasetid]
        ### sequence data file-name [protein && rna]
        self.seq_datasets={}
        
        self.seq_datasets[ self.dataset_suit[0] ]={}
        self.seq_datasets[ self.dataset_suit[0] ]['protein']='NPInter10412_Pall'
        self.seq_datasets[ self.dataset_suit[0] ]['rna']='NPInter10412_linearfold_bpRNA'
        
        self.seq_datasets[ self.dataset_suit[1] ]={}
        self.seq_datasets[ self.dataset_suit[1] ]['protein']='reRPI2825_Pall'
        self.seq_datasets[ self.dataset_suit[1] ]['rna']='reRPI2825_linearfold_bpRNA'
        
        self.seq_datasets[ self.dataset_suit[2] ]={}
        self.seq_datasets[ self.dataset_suit[2] ]['protein']='RPI488_Pall'
        self.seq_datasets[ self.dataset_suit[2] ]['rna']='RPI488_linearfold_bpRNA'
        
        self.seq_datasets[ self.dataset_suit[3] ]={}
        self.seq_datasets[ self.dataset_suit[3] ]['protein']='RPI2241_Pall'
        self.seq_datasets[ self.dataset_suit[3] ]['rna']='RPI2241_linearfold_bpRNA'
        
        self.seq_datasets[ self.dataset_suit[4] ]={}
        self.seq_datasets[ self.dataset_suit[4] ]['protein']='RPI1807_Pall'
        self.seq_datasets[ self.dataset_suit[4] ]['rna']='RPI1807_linearfold_bpRNA'

        self.seq_datasets[ self.dataset_suit[5] ]={}
        self.seq_datasets[ self.dataset_suit[5] ]['protein']='LPI43250_Pall'
        self.seq_datasets[ self.dataset_suit[5] ]['rna']='LPI43250_rna17672_linearfold_bpRNA'
        
        self.seq_datasets[ self.dataset_suit[6] ]={}
        self.seq_datasets[ self.dataset_suit[6] ]['protein']='EVLncRNAs_protein45_Pall'
        self.seq_datasets[ self.dataset_suit[6] ]['rna']='EVLncRNAs_rna34_linearfold_bpRNAst'

        ### pair file-name
        self.positive_pair_file={}
        self.positive_pair_file[ self.dataset_suit[0] ]=['NPInter10412_pos_pairs.txt']
        self.positive_pair_file[ self.dataset_suit[1] ]=['RPI2825_pos_pairs.txt','RPI390_pos_pairs.txt']
        self.positive_pair_file[ self.dataset_suit[2] ]=['RPI488_pairs.txt']
        self.positive_pair_file[ self.dataset_suit[3] ]=['RPI2241_pos_pairs.txt','RPI369_pos_pairs.txt']
        self.positive_pair_file[ self.dataset_suit[4] ]=['RPI1807_pairs.txt']
        self.positive_pair_file[ self.dataset_suit[5] ]=['LPI43250_pos_pairs.txt']
        self.positive_pair_file[ self.dataset_suit[6] ]=['EVLncRNAs_pos_pairs.txt']
        
        ### cluster file-name
        self.cluster_file_root=''
        self.cluster_file={}
        self.cluster_file[ self.dataset_suit[0] ]={}
        self.cluster_file[ self.dataset_suit[0] ]['protein']='NPinter_protein40%_cdhit4.8.1.clstr'
        self.cluster_file[ self.dataset_suit[0] ]['rna']='NPinter_RNA80%_cdhit4.8.1.clstr'
        
        self.cluster_file[ self.dataset_suit[1] ]={}
        self.cluster_file[ self.dataset_suit[1] ]['protein']='reRPI2825_protein40%_cdhit4.8.1.clstr'
        self.cluster_file[ self.dataset_suit[1] ]['rna']='reRPI2825_RNA80%_cdhit4.8.1.clstr'
        
        self.cluster_file[ self.dataset_suit[3] ]={}
        self.cluster_file[ self.dataset_suit[3] ]['protein']='RPI2241_positive_protein40%_cdhit4.8.1.clstr'
        self.cluster_file[ self.dataset_suit[3] ]['rna']='RPI2241_positive_RNA80%_cdhit4.8.1.clstr'
        
        self.cluster_file[ self.dataset_suit[5] ]={}
        self.cluster_file[ self.dataset_suit[5] ]['protein']='LPI43250_protein56_40%_cdhit4.8.1.clstr'
        self.cluster_file[ self.dataset_suit[5] ]['rna']='LPI43250_rna17672_80%_cdhit4.8.1.clstr'
        
        return
    
    
    def getDataPath(self,data_id=None):

        self.fileroot = os.path.join(self.fileroot,self.dataset_suit[ data_id ])
        path_seq_folder_protein = os.path.join(self.fileroot,
                                               self.seq_datasets[ 
                                                   self.dataset_suit[ data_id ]
                                               ]['protein']
                                              )
        path_seq_folder_rna = os.path.join(self.fileroot,
                                           self.seq_datasets[ 
                                               self.dataset_suit[ data_id ]
                                           ]['rna']
                                          )
        return {'protein':path_seq_folder_protein,'rna':path_seq_folder_rna}

dataPathLoader(0).getDataPath()
