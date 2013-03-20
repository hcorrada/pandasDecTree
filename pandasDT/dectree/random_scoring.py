import numpy as np
from Node import Split

def gini(p): return np.sum(p*(1-p))

def scoreit(left_p, right_p, nleft, nright, fn=gini):
    n = float(nleft + nright)
    fleft = fn(left_p)
    fright = fn(right_p)
    return nleft/n * fleft + nright/n * fright

def score_feature_value(df, feature_name, feature_value, label_name):
    cnts = df.groupby([feature_name, label_name]).size()

    feature_values, class_labels = cnts.index.levels
    if (len(feature_values)==1):
        return np.inf

    feature_index, label_index = cnts.index.labels

    get_props = lambda indx: np.array([sum(cnts[indx][label_index[indx]==j]) for j in range(len(class_labels))])/float(sum(cnts[indx]))

    i = np.where(feature_values == feature_value)[0]
    indx = np.where(feature_index == i)[0]

    left_props = get_props(indx)
    nleft = sum(cnts[indx])

    indx = feature_index != i
    right_props = get_props(indx)
    nright = sum(cnts)-nleft
    res = scoreit(left_props, right_props, nleft, nright)
    return res

def get_candidate_values(df, feature_name):
    cnts = df.groupby([feature_name]).size()
    if (len(cnts)==1):
        return []

    feature_values = cnts.index.values
    return [(feature_name, feature_value) for feature_value in feature_values]

def best_split(df, features_to_use, label_name, nfeatures_per_split):
    feature_names = df.columns.values[features_to_use]
    candidate_values = []
    for feature_name in feature_names:
        candidate_values += get_candidate_values(df, feature_name)

    if (len(candidate_values)==0):
        return None, None, None

    if len(candidate_values)>nfeatures_per_split:
        indx=np.arange(len(candidate_values))
        indx=np.random.permutation(indx)[:nfeatures_per_split]
    else:
        indx=np.arange(len(candidate_values))

    scores = np.inf * np.ones(len(indx))
    for i in range(len(indx)):
        j = indx[i]
        feature_name = candidate_values[j][0]
        feature_value = candidate_values[j][1]
        res = score_feature_value(df, feature_name, feature_value, label_name)
        #print res
        scores[i] = res

    #print 'Best feature scores: ', scores
    #print scores
    minIndx = np.argmin(scores)
    return Split(candidate_values[minIndx][0], candidate_values[minIndx][1]), features_to_use, scores[minIndx]

