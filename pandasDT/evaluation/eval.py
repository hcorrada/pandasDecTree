import pandas as pd
import numpy as np

def recall(preds, labels):
    df=pd.DataFrame({'pred':preds,'label':labels})
    mat=df.groupby(['label','pred']).size()
    try:
        npos = mat[1]
    except:
        return 1.

    try:
        return npos[1]/float(sum(npos))
    except:
        return 0.

def precision(preds, labels):
    df=pd.DataFrame({'pred':preds,'label':labels})
    mat=df.groupby(['pred','label']).size()
    try:
        npos=mat[1]
    except:
        return 0.

    try:
        return npos[1]/float(sum(npos))
    except:
        return 1.

def fscore(preds, labels):
    rec = recall(preds,labels)
    prec = precision(preds,labels)
    print rec, prec
    return (2*rec*prec)/(rec+prec)

def error_rate(preds, labels):
    return np.mean(preds != labels)

def weighted_error_rate(preds, labels, weights):
    return np.sum(weights[preds != labels])/float(np.sum(weights))


