import pandas as pd 
import numpy as np 
import collections

def loaddata(path='arrhythmia/arrhythmia.data'):
    df = pd.read_csv(path, header=None)
    df = df.dropna(axis=1)
    
    X = df.iloc[:,0:-1].values
    Y = df.iloc[:,-1].values
    
    all_classes = np.unique(Y)
    anom_classes = np.asarray([3,4,5,7,8,9,14,15])
    reg_classes = np.setdiff1d(all_classes, anom_classes)
    anom_ind = [i for i,val in enumerate(Y) if val in anom_classes]
    reg_ind = [i for i,val in enumerate(Y) if val in reg_classes]
    Y[anom_ind] = 0
    Y[reg_ind] = 1
    print('X: {}, counts: {} (ratio: {})'.format(X.shape, collections.Counter(Y), len(anom_ind)/(len(anom_ind) + len(reg_ind))))    
    return X, Y, np.asarray(reg_ind), np.asarray(anom_ind)