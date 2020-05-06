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
#%%
### global control ###
max_datarecord=500
tot_partition_num=150
PARTITIONS=50
K=200
log_path='//outputlog_spark.txt'

#%%
### debug function###
class Debug:
    mode=1
    def DEBUG(self,x):
        if self.mode==1:
            print('DEBUG: '+str(x))
        return
    def DEBUGF(self,x='\n'):
        if self.mode==0:
            return
        with open(log_path,'a+') as f:
            if type(x)==list:
                for line in x:
                    f.write(str(line))
                    f.write('\n')
                    print(' DEBUG: '+str(line))
            else:
                f.write(str(x))
                f.write('\n')
                print(' DEBUG: '+str(x))
        return
    def DelFILE(self):
        if self.mode==0:
            return
        if os.path.exists(log_path):
            os.path.os.remove(log_path)
        return
dbg=Debug()
dbg.mode=2
dbg.DelFILE()

'''
    debug 0
    info 1
    err 2
'''
def INFO(msg):
    try:
        if dbg.mode>=1:
            print('INFO::'+msg)
    except Exception as e:
        print(e)
    return

def ERR(msg):
    try:
        if dbg.mode>=2:
            print('ERR::'+msg)
    except Exception as e:
        print(e)
    return

def flogging(msg):
    try:
        with open('./logging.txt','a+') as f:
            f.write(str(time.localtime())+'>\t'+msg+'\n')
    except Exception as e:
        ERR(e.args)
    return

#### data path configuration ####
class dataPathLoader:
    data_cluster_root = 'Negative1'
#     fileroot = r'D:\StatisticsData\BIODATA\Data'
    fileroot = r'D:\StatisticsData\BIODATA\Data\Data'
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
        
        return
    
    def getDataPath(self,data_id=None):
        if data_id:
            data_id = data_id
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


#%%
### load dataset ###
class dataLoader:
    ### suite 1 for RPI
    ### suite 2 for NPinter
   
    K=3
    
    dataset_cluster_root='Negative1'
    
    #### cluster data root ####
    # fileroot='//home/ossfsData'
    #### local data root ####
#     fileroot='D:\\StatisticsData\\BIODATA\\Data'
    fileroot = r'D:\StatisticsData\BIODATA\Data\Data'
    
    hasNegative=[0,0,1,0,1,0,1]
    
    def __init__(self,datasetid,K=3):
        self.dataset_id=datasetid
        self.K=K
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
    
    
    ''' read the protein seq from file
        return @{protein_id:[   [protein_info]  ]}
    '''
    def readSequenceProtein(self):
        seq_protein_array={}
        foldername=self.seq_datasets[self.datasetname]['protein']
        filedir=os.path.join(os.path.join(self.fileroot,self.datasetname),foldername)
        for root,director,files in os.walk(filedir):
            for fname in files:
                proteinInfo=[]
                openfilepath=os.path.join(root,fname)
                with open(openfilepath,'r') as f:
                    text=f.read()
                    content_protein=text.split('\n')[0:6] ### 文本需求？
                    content_protein[0] = content_protein[0].lstrip('>')
                    proteinInfo=content_protein
                if proteinInfo[0] in seq_protein_array.keys():
                    print('ERROR:: duplicate protein when reading seq files ',proteinInfo[0])
                else:
                    seq_protein_array[proteinInfo[0]]=proteinInfo
        return seq_protein_array
    
    def readSequenceRna(self):
        seq_rna_array={}
        foldername=self.seq_datasets[self.datasetname]['rna']
        filedir=os.path.join(os.path.join(self.fileroot,self.datasetname),foldername)
        for root,director,files in os.walk(filedir):
            for fname in files:
                rnaInfo=[]
                openfilepath=os.path.join(root,fname)
                with open(openfilepath,'r') as f:
                    text=f.read()
                    content_rna=text.split('\n')[0:7] 
                    content_rna[0] = re.split('#Name: ',content_rna[0])[1]
                    [content_rna.pop(1) for i in range(2)]
                    rnaInfo=content_rna
                if rnaInfo[0] in seq_rna_array.keys():
                    print('ERROR:: duplicate rna when reading seq files ',rnaInfo[0])
                else:
                    seq_rna_array[rnaInfo[0]]=rnaInfo
                
        return seq_rna_array
    
    ''' get the protein & rna clusters from file
        return @ [] null list for has_negative
               @clusters_protein {protein_id:cluster_id}
               @clusters_rna {rna_id:cluster_id}
    '''
    def readCluster(self):
        if self.hasNegative[self.dataset_id]:
            return [{},{}]
        else:
            dir_cluster=os.path.join(self.fileroot,self.dataset_cluster_root)
            ### concatenate protein cluster file path
            protein_fname_cluster=self.cluster_file[self.datasetname]['protein']
            protein_path_cluster=os.path.join(dir_cluster,protein_fname_cluster)
            ###### the code for protein and rna cluster process is the same,
            ###### so maintain them at the same time
            ### get the protein clusters from file
            clusters_protein={}
            with open(protein_path_cluster,'r') as f:
                text=f.read()
                content_cluster=text.split('\n')
                lastgroupid=-1
                for line in content_cluster:
                    if '>Cluster' in line:
                        lastgroupid=re.search('[0-9]+',line)[0]
                    else:
                        reg_result = re.findall('>(.+)\.\.\.',line)
                        if len(reg_result):
                            name = reg_result[0]
                            if name not in clusters_protein.keys():### repetition may cause override
                                clusters_protein[name]=lastgroupid
                            else:print('error protein recorded in cluster '+str(name))
                        else:
                            print('ERROR:: regex ',line)
            
            ### get the rna cluster file path
            rna_fname_cluster=self.cluster_file[self.datasetname]['rna']
            rna_path_cluster=os.path.join(dir_cluster,rna_fname_cluster)
        
            ### get the rna clusters from file
            clusters_rna={}
            with open(rna_path_cluster,'r') as f:
                text=f.read()
                content_cluster=text.split('\n')
                lastgroupid=-1
                for line in content_cluster:
                    if '>Cluster' in line:
                        lastgroupid=re.search('[0-9]+',line)[0]
                    else:
                        reg_result = re.findall('>(.+)\.\.\.',line)
                        if len(reg_result):
                            name = reg_result[0]
                            if name not in clusters_rna.keys():### repetition may cause override
                                clusters_rna[name]=lastgroupid
                            else:print('error rna recorded in cluster '+str(name))
                        else:
                            print('ERROR:: regex ',line)
                            
        return [clusters_protein,clusters_rna]
    
    ''' read the pair from file
        parameter @clusters_protein & @clusters_rna are return value of function @readCluster
        return  @pairs {protein_id,rna_id:0/1}
                @self.group_pairs {cluster_of_protein,cluster_of_rna:0/1}
    '''
    def readPair(self,clusters_protein,clusters_rna):
        fpath=os.path.join(self.fileroot,self.datasetname)
        fnames=self.positive_pair_file[self.datasetname]
        pairs={}
        group_pairs={}
        if self.hasNegative[self.dataset_id]==1:
            for fname in fnames:
                with open(os.path.join(fpath,fname), 'r') as f:
                    text=f.read()
                    content=text.split('\n')
                    for line in content:
                        if len(line):
                            items=line.split('\t')
                            try:
                                pairs[items[0]+','+items[1]]=int(items[2])
                            except Exception as e:
                                print('items assignment exception '+str(items))
                        else: 
                            print('read line error '+line)  
        else:
            for fname in fnames:
                ### all pairs in file is positive
                with open(os.path.join(fpath,fname), 'r') as f:
                    text=f.read()
                    content=text.split('\n')
                    content.pop(0) ### eliminate the header
                    for line in content:
                        if len(line):
                            items=line.split('\t')
                            try:
                                pairs[items[0]+','+items[1]]=1
                                if clusters_protein[items[0]]+','+clusters_rna[items[1]] not in group_pairs.keys():
                                    group_pairs[clusters_protein[items[0]]+','+clusters_rna[items[1]]]=1
                                else:
                                    # print('key exists')
                                    pass
                            except Exception as e:
                                print('items decode exception '+items)
                        else: 
                            print('read line error '+line) 
        self.group_pairs=group_pairs
        return pairs
    
    ''' randomly generate Negative pair from sequence read by function @readSequenceProtein
        and @protein_seq & @rna_seq are parameters, while N controls the number
        and @protein_cluster & @rna_cluster are cluster dictionary generated by function @readCluster
        return @negative_pairs {protein_id,rna_id:0}
    '''
    def generateNegativePair(self,protein_seq,rna_seq,protein_cluster,rna_cluster,N):
        if self.hasNegative[self.dataset_id]==1:
            return {}
        else:
            negative_pairs={}
            count=0
            negative_pair_paths = './data/negativePairs'+str(self.dataset_id)+'.txt'
            
            def generatePairs(count,negative_pairs):
                protein_keys=list(protein_seq.keys())
                rna_keys=list(rna_seq.keys())
                stag=0
                start_time = time.clock()
                timeout = 120 ### 超时时间
                while count<N and stag<N and time.clock()-start_time<timeout:
                    proteinidx=random.randint(0,len(protein_seq)-1)
                    rnaidx=random.randint(0,len(rna_seq)-1)
                    rand_protein=protein_keys[proteinidx]
                    rand_rna=rna_keys[rnaidx]
                    ### pair not already exists in negative-pairs && cluster-ids not already exists in self.group-pairs
                    if rand_protein+','+rand_rna not in negative_pairs.keys() and \
                    rand_rna+','+rand_protein not in negative_pairs.keys() and \
                    rand_protein in protein_cluster and \
                    rand_rna in rna_cluster and \
                    protein_cluster[rand_protein]+','+rna_cluster[rand_rna] not in self.group_pairs.keys():
                        self.group_pairs[protein_cluster[rand_protein]+','+rna_cluster[rand_rna]]=0 ### update group-pairs
                        negative_pairs[rand_protein+','+rand_rna]=0
                        stag=0
                    else:
                        if (rand_rna not in rna_cluster) or (rand_protein not in protein_cluster):
                            print('ERR::error seq not in cluster') 
                        stag=stag+1
                        continue
                    count=count+1
                return [count,negative_pairs]
            
            def writePairsGenerateNegativePair(count,negative_pairs):
                text = ''
                for line in negative_pairs.keys():
                    text += line +'\n'
                with open(negative_pair_paths,'w+') as f:
                    f.write(text)
                return
            
            def readPairsGenerateNegativePair():
                with open(negative_pair_paths,'r+') as f:
                    text = f.read()
                    lines = text.strip('\n').split('\n')
                    negative_pairs = dict(list(map(lambda x:(x,0),lines)))
                    count = len(negative_pairs)
                    INFO('count of negative pairs'+str(count))
                if len(negative_pairs)==0:
                    ERR('negative pairs empty')
                return [count,negative_pairs]
            
            if os.path.exists(negative_pair_paths):
                [count,negative_pairs] = readPairsGenerateNegativePair()
            else:
                [count,negative_pairs] = generatePairs(count,negative_pairs)
                writePairsGenerateNegativePair(count,negative_pairs)
            return negative_pairs
    
    ''' temporary function should then be merged into @main
    '''
    def getPairs(self,Nnegative):
        P=dl.readSequenceProtein()
        R=dl.readSequenceRna()
        [Pc,Rc]=dl.readCluster()
        pairs=dl.readPair(Pc,Rc)
        negative_pairs={}
        if self.hasNegative[self.dataset_id]==0:
            negative_pairs=dl.generateNegativePair(P,R,Pc,Rc,Nnegative)
        if len(set(pairs.keys()).intersection(set(negative_pairs.keys())))!=0:
            print('error positive and negative pairs overlap')
        pairs={**pairs,**negative_pairs}
        # for key in negative_pairs.keys():
        #     pairs[key]=negative_pairs[key]
        return pairs
    
    ''' stage 1 process single <em> protein </em> sequence, convert the seq into encoded seq
        parameter @singleSeqProtein single dict_item from @protein-sequence 
        return @feature {protein-id:[   [extracted-protein-features]  ]}
    '''
    def processProteinSequence(self,singleSeqProtein):
        seqlen=len(singleSeqProtein[1])
        # filter the sequences
        structureclass={'R':0,'K':0,'H':0,
                        'N':1,'W':1,'S':1,'Q':1,'Y':1,'G':1,'T':1,
                        'P':2,'M':2,'F':2,'D':2,'A':2,'V':2,'L':2,'I':2,
                        'C':3,'E':3}
        singleSeqProtein[1] = singleSeqProtein[1].lstrip('X')
        newseq=''
        for item in singleSeqProtein[1]:
            if item not in structureclass.keys():
                print('structure decode error '+item)
                return {}
            newseq=newseq+str(structureclass[item])
        singleSeqProtein[1]=newseq
        # mask 2ed with 6th line
        newstructure=''
        for i in range(seqlen):
            if singleSeqProtein[5][i]=='*' and singleSeqProtein[2][i]=='C':
                newstructure=newstructure+'*'
            else:
                newstructure=newstructure+singleSeqProtein[2][i]
        # process
        seq=newstructure
        feature=[]
        s=0
        e=0
        while s<seqlen:
            if e<seqlen and seq[s]==seq[e]:
                e=e+1
            else:
                feature.append([singleSeqProtein[1][s:e],newstructure[s:e],
                                singleSeqProtein[3][s:e],singleSeqProtein[4][s:e]])
                if e<seqlen:
                    s=e
                else:
                    break
        # debug format
        # feature=np.array(feature)
        feature={singleSeqProtein[0]:feature}
        return feature
    
    ''' stage 1 process single <em> rna </em> sequence, convert the seq into encoded seq
        parametesr @singleSeqRna single dict_item from @Rna-sequence 
        return @feature {rna-id:[   [extracted-rna-features]  ]}
    '''
    def processRNASequence(self,singleSeqRna):
        seqlen=len(singleSeqRna[1])
        seq=singleSeqRna[3]
        feature=[]
        # filter the sequence
        structureclass={'A':0,'U':1,'G':2,'C':3}
        singleSeqRna[1] = singleSeqRna[1].lstrip('X')
        newseq=''
        for item in singleSeqRna[1]:
            if item not in structureclass.keys():
                print('structure decode error '+item)
                return {}
            newseq=newseq+str(structureclass[item])
        singleSeqRna[1]=newseq
        # process    
        s=0
        e=0
        while s<seqlen:
            if e<seqlen and seq[s]==seq[e]:
                e=e+1
            else:
                feature.append([singleSeqRna[1][s:e],singleSeqRna[2][s:e],
                                singleSeqRna[3][s:e],singleSeqRna[4][s:e]])
                if e<seqlen:
                    s=e
                else:
                    break
        feature={singleSeqRna[0]:feature}
        return feature
    
    ''' k-mer split protein features and merge them into one dict
        parameter @features combined as a entire features dict each of which 
        is returned by function @processProteinSequence
        [   id  , array([ [split-features] ]) ] 
        but pass in 
        {   id  :  [    [split-features]  ] }
        
        return @newfeatures(split-features) { sequence-id : { feature-pattern : frequency } }
               @tot_features set of all features
    '''
    def KmerSplitProtein(self,features,k=3):
        newfeatures={}
        totfeatures=[]
        for seqname in features:
            if len(features[seqname])==0: # added condition for filtered sample
                print('error null features '+seqname)
                continue
            seqfeatures={}
            for f in features[seqname]:
                L=len(f[0])
                for i in range(L-k+1):
                    Mct=f[3][i:i+k].count('M')
                    Bct=f[3][i:i+k].count('B')
                    Ect=f[3][i:i+k].count('E')
                    # biological method
                    key=str(str(f[1][0])+','+str(f[0][i:i+k])+
                            ','+str(Mct)+','+str(Bct)+','+str(Ect))
                    # naive method
                    # key=str(str(f[0][i:i+k]))
                    if key in seqfeatures.keys():
                        seqfeatures[key]=seqfeatures[key]+1
                    else:
                        seqfeatures[key]=1
                    totfeatures.append(key)
            if seqname in newfeatures.keys():
                print('error sequence features already exists '+seqname)
            newfeatures[seqname]=seqfeatures
        totfeatures=set(totfeatures)
        return newfeatures,totfeatures

    def KmerSplitRna(self,features,k=3):
        newfeatures={}
        totfeatures=[]
        for seqname in features:
            if len(features[seqname])==0: # added condition for filtered sample
                print('error null features '+seqname)
                continue
            seqfeatures={}
            for f in features[seqname]:
                L=len(f[0])
                for i in range(L-k+1):
                    # biological method
                    key=str(f[2][0])+','+str(f[0][i:i+k])
                    # naive method
                    # key=str(f[0][i:i+k])
                    if key in seqfeatures.keys():
                        seqfeatures[key]=seqfeatures[key]+1
                    else:
                        seqfeatures[key]=1
                    totfeatures.append(key)
            if seqname in newfeatures.keys():
                print('error sequence features already exists '+seqname)
            newfeatures[seqname]=seqfeatures
        totfeatures=set(totfeatures)
        return newfeatures,totfeatures
    
    ''' transform the dictionary data into matrix
        parameter   @total_features all features set
                    @all_pairs total pairs containing postive and negative ones
                    @protein_features split-protein-features computed by function @KmerSplitProtein
                    @rna_features split-rna-features computed by function @KmerSplitRna
                    @flag ['train','test']
        return      @dataRDD spark.RDD  [ (index , Row(features=Vectors.dense()) , Row(label=num)  ) ]
    '''
    def matrixTransformation(self,total_features,all_pairs,protein_features,rna_features,flag='train'):
        total_features=np.array(list(total_features)) ### make dict to accelerate
        print('DEBUG:: total features count ',str(len(total_features)))
        
        tf_dict={} ### record {feature : feature idx}
        for i,line in zip(range(len(total_features)),total_features):
            tf_dict[line]=i
#         print(all_pairs)
        X=[]
        Y=[]
        for i,pair in enumerate(all_pairs):
            item=pair.split(',')
            if len(item)!=2 or len(pair)<3:
                print('error pair number is not 2 '+item)
                continue
                
            y=all_pairs[pair]
            
            try:
                pfeature=protein_features[item[0]]
            except Exception as e:
                print('ERROR:: protein ',e,item[0])
                
            try:
                rfeature=rna_features[item[1]]
            except Exception as e:
                print('ERROR:: rna ',e,item[1])
                
            x = {}
            for pf in pfeature:
                try:
                    idx=tf_dict[pf] ### originally tf.index 
                    x[idx]=pfeature[pf]
                except Exception as e:
                    print('error protein feature not in tot-features '+pf)
            for rf in rfeature:
                try:
                    idx=tf_dict[rf] ### originally tf.index
                    x[idx]=rfeature[rf]
                except Exception as e:
                    print('error rna feature not in tot-features '+rf)
            if y is None:
                print('error target invalid '+pair+'\t'+str(all_pairs[pair]))
                continue
            X.append(x)
            Y.append(y)
        return [[X,Y],len(total_features)]
    
    
def Dict2Matrix(idx, k, data):
    N = len(data[0][0])
    Dim = data[1]
    mat = []
    df = pd.DataFrame()
    for i in range(N):
#         print(i)
        vec = [0 for j in range(Dim)]
        for (k,v) in data[0][0][i].items():
            vec[k] = v
        vec.extend([data[0][1][i]])
        mat.append(vec)
        
        if i%2000==1999:
            df = df.append(pd.DataFrame(np.array(mat)))
            mat = []
    if len(mat):
        df = df.append(pd.DataFrame(np.array(mat)))
        mat = []
    return df

from sklearn.decomposition import TruncatedSVD

def Dict2Sparse(data):
    row = []
    col = []
    entry = []
    Len = len(data[0][0])
    Dim = data[1]
    for i in range(Len):
        c = []
        e = []
        for (k,v) in data[0][0][i].items():
            c.append(k)
            e.append(v)
        r = [i for j in range(len(c))]
        row.extend(r)
        col.extend(c)
        entry.extend(e)
    res = [row, col, entry, Len, Dim, data[0][1]]
    return res

def outputMatrix(idx,k,data):
#    f=open('E:/out'+str(idx)+'_'+str(k)+'.txt','a+')
    print('data',len(data[0][0]),'positive',sum(np.array(data[0][1])==1))
    with open('E:/out'+str(idx)+'_'+str(k)+'.txt','a+') as f:
        for i in range(len(data[0][0])):
            line =''
            print('write line',i)
            x = data[0][0][i]
            y = data[0][1][i]
            xx = [str(0) for i in range(data[1])]
            for (k,v) in x.items():
                xx[k] = str(v)
            del x
            x = xx
            del xx
            line += ','.join(x) + ',' +str(y) +'\n'
            f.write(line)
#    f.close()
    return

class trainer:
    dls=[]
    testdl=[]
    pairs={}
    features_set=set([])
    protein_features={}
    rna_features={}
    test_ratio=0.3
    local_partition=20
    model_path='D://StatisticsData//BIODATA//model'
    
    '''
        n=5, k=-1
    '''
    def __init__(self,n,k):
        if k!=-1:
#             for i in set(range(n))-set(list(k)):
#                 self.dls.append(dataLoader(i))
#             self.testdl=dataLoader(k)
            self.dls.append(dataLoader(n))
        else:
            for i in range(n):
                self.dls.append(dataLoader(i))
        return
    
    ''' 
        train
    '''
    def train(self,dataRDD,testdataRDD):
        
        return

    def MAIN_SINGLE_TEST(self,sidx,pk=6,rk=6):
        print('=============RETRIEVE TRIAN DATA=================')
        suit = 0
        for dl in self.dls:
            if sidx == suit:
                print('# DEBUG: # DEBUG: **************new dl %d***************'%dl.dataset_id)
                print('# DEBUG: READ SEQ FROM FILE')
                P=dl.readSequenceProtein()
                R=dl.readSequenceRna()

                print('# DEBUG: READ CLUSTER FROM FILE')
                [Pc,Rc]=dl.readCluster()

                print('# DEBUG: READ PAIR FROM FILE')
                pairs=dl.readPair(Pc,Rc)

                print('# DEBUG: GENERATE NEGATIVE PAIR')
                print('# DEBUG: negative pair number',len(pairs))
                negative_pairs=dl.generateNegativePair(P,R,Pc,Rc,len(pairs))
                all_pairs={**pairs,**negative_pairs}

                print('# DEBUG: PAIR UNION')
                self.pairs={**self.pairs,**all_pairs}

                print('# DEBUG: EXTRACT FEATURES--PROTEIN')
                protein_features={}
                for single_protein in P:
                    single_protein_features=dl.processProteinSequence(P[single_protein])
                    item=single_protein_features.popitem()
                    protein_features[item[0]]=item[1]

                print('# DEBUG: EXTRACT FEATURES--RNA')
                rna_features={}
                for single_rna in R:
                    single_rna_features=dl.processRNASequence(R[single_rna])
                    item=single_rna_features.popitem()
                    rna_features[item[0]]=item[1]

                print('# DEBUG: K-MER CALCULATION')
                [split_protein_features,total_split_protein_features]=dl.KmerSplitProtein(protein_features,pk)
                [split_rna_features,total_split_rna_features]=dl.KmerSplitRna(rna_features,rk)

                print('# DEBUG: FEATURE UNION')
                self.features_set=self.features_set.union( total_split_rna_features.union(total_split_protein_features) )
                self.protein_features={**self.protein_features,**split_protein_features}
                self.rna_features={**self.rna_features,**split_rna_features}

                print('# DEBUG: GARBAGE COLLECTION')
                del P,R
                del Pc,Rc
                print('MATRIX TRANSFORMATION')
                data=dl.matrixTransformation(total_split_rna_features.union(total_split_protein_features)
                                                ,all_pairs
                                                ,split_protein_features
                                                ,split_rna_features
                                                ,'train')
#                 print(len(total_split_rna_features),len(total_split_protein_features))
                data.extend([len(total_split_protein_features),len(total_split_rna_features)])
                return data
                # outputMatrix(suit,6,data)
                # df = Dict2Matrix(suit,6,data)
#                 data = Dict2Sparse(data)
                print('data',len(data[0][0]),'positive',sum(np.array(data[0][1])==1))
                print('TRAINING')
                self.train(data,None)
                print('GARBAGE COLLECTION')
                del split_protein_features,split_rna_features
                del all_pairs
                del data
            suit += 1
        return
    
    
    

    
# ========================= read data from other sources ================
DATA_ROOT = 'D:/MYPROJECT/RPITER/data/'
DATA_SET = 'RPI488'

path = os.path.join(DATA_ROOT,DATA_SET+'_pairs.txt')


### read pair data of one dataset
def importPairs():
    with open(path,'r+') as f:
        text = f.read()
        content = text.split('\n')
        res = set()
        for line in content:
            items = line.split('\t')
            if len(items)>=2:
                res.add(items[0])
                res.add(items[1])
        res = list(res)