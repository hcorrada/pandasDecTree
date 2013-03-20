import numpy as np
from Node import Split

def gini(p): return np.sum(p*(1-p))

def scoreit(left_p, right_p, nleft, nright, fn=gini):
    n = float(nleft + nright)
    fleft = fn(left_p)
    fright = fn(right_p)
    return nleft/n * fleft + nright/n * fright

def score_feature(df, feature_name, label_name, importance):
    curdf=df[[feature_name,label_name]]
    curdf['importance']=importance

    cnts = curdf.groupby([feature_name, label_name]).sum()['importance']

    feature_values, class_labels = cnts.index.levels
    if (len(feature_values)==1):
        return feature_values.values, [np.inf]

    feature_index, label_index = cnts.index.labels

    get_props = lambda indx: np.array([sum(cnts[indx][label_index[indx]==j]) for j in range(len(class_labels))])/float(sum(cnts[indx]))

    score = np.empty(len(feature_values))

    for i in range(len(feature_values)):
        indx = feature_index == i
        left_props = get_props(indx)
        nleft = sum(cnts[indx])

        indx = feature_index != i
        right_props = get_props(indx)
        nright = sum(cnts)-nleft
        score[i] = scoreit(left_props, right_props, nleft, nright)
    return feature_values.values, score

def best_feature_value(df, feature_name, label_name, importance):
    #print 'Scoring feature %s' % feature_name,
    score = score_feature(df, feature_name, label_name, importance)
    #print score
    i = np.argmin(score[1])
    return (score[0][i], score[1][i])

def best_split(df, features_to_use, label_name, importance):
    feature_names = df.columns.values[features_to_use]
    best_values = [best_feature_value(df, feature_name, label_name, importance) for feature_name in feature_names]

    scores = np.array([score for (value,score) in best_values])
    #print 'Best feature scores: ', scores
    minIndx = np.argmin(scores)
    features_left = np.delete(features_to_use, minIndx)
    return Split(feature_names[minIndx], best_values[minIndx][0]), features_left, scores[minIndx]

