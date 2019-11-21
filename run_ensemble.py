import numpy as np
import pandas as pd
import dataloader as dl
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, precision_recall_fscore_support
import copy
import collections

REG_LAB = 0
ANOM_LAB = 1

params = dict(
    {
        'n_estimators': 1000,
        'max_samples': 0.5,
        'max_features': 0.8,
        'n_jobs': 20,
        'trainfrac': 0.5,
        'sim_reps': 2,
        'score_thresh': 0.5
    }
)

X, Y, reg_ind, anom_ind = dl.loaddata()

iforest = IsolationForest(
    n_estimators=params['n_estimators'] ,
    max_samples=params['max_samples'],
    max_features=params['max_features'],
    n_jobs=params['n_jobs'],
    behaviour='new',
    contamination=0.001)

ocsvm = OneClassSVM(gamma='scale', nu=0.05)

def traintest(model, trainX, trainY, testX, testY):
    all_metrics = dict({'f1':[], 'precision':[], 'recall':[], 'mcc':[]})
    model.fit(trainX)
    train_scores = model.score_samples(trainX)
    thresh = np.percentile(train_scores, 5)
    scores = model.score_samples(testX)
    predY = np.where(scores <= thresh, ANOM_LAB, REG_LAB)
    prec, recall, f1, _ = precision_recall_fscore_support(testY, predY, average="binary", pos_label=ANOM_LAB)
    all_metrics['f1'] = f1
    all_metrics['precision'] = prec
    all_metrics['recall'] = recall
    all_metrics['mcc'] = matthews_corrcoef(testY, predY)
    print('f1 = {:.3} (pr={:.3}, re={:.3}), mcc = {:.3}'.format(
        all_metrics['f1'],
        all_metrics['precision'],
        all_metrics['recall'],
        all_metrics['mcc']))
    return all_metrics

iforest_metrics = []
svm_metrics = []
for i in range(0, params['sim_reps']):
    reg_sufffled = copy.deepcopy(reg_ind)
    np.random.shuffle(reg_sufffled)
    tr_ind = reg_sufffled[0:int(Y.size*params['trainfrac'])].ravel()
    te_ind = reg_sufffled[int(Y.size*params['trainfrac']):].ravel()
    trainX = X[tr_ind,:]
    trainY = REG_LAB * np.ones((tr_ind.size,1))
    testX = np.vstack((X[te_ind,:], X[anom_ind,:]))
    testY = np.vstack((REG_LAB * np.ones((te_ind.size,1)), ANOM_LAB * np.ones((anom_ind.size,1))))
    print('train: {}({}), test: {} ({})'.format(trainX.shape, collections.Counter(trainY.ravel()), testX.shape, collections.Counter(testY.ravel())))
    print('----------IFOREST--------------')
    iforest_metrics.append(traintest(iforest, trainX, trainY, testX, testY))
    print('---------OCSVM--------------')
    svm_metrics.append(traintest(ocsvm, trainX, trainY, testX, testY))

df_iforest_met = pd.DataFrame(iforest_metrics)
df_ocsvm_met = pd.DataFrame(svm_metrics)

print('-- Average IFOREST metrics')
print(df_iforest_met.mean())
print('-- Average OCSVM metrics')
print(df_ocsvm_met.mean())