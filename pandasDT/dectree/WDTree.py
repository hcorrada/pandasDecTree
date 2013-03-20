from weighted_scoring import best_split
from DTree import DTree
import numpy as np

class WDTree(DTree):
    def __init__(self, traindat, label_name, importance):
        super(WDTree, self).__init__(traindat, label_name)
        self.importance = importance

    def get_class_props(self, indxs):
        curdf=self.traindat.ix[indxs,:]
        curdf['importance']=self.importance[indxs]

        cnts = curdf.groupby(self.label_name).sum()['importance']
        props = cnts/float(np.sum(cnts))
        class_labels = cnts.index
        return (dict(zip(class_labels, props)))

    def split_func(self, indxs, features_to_use):
        return best_split(self.traindat.ix[indxs,:], features_to_use, self.label_name, self.importance[indxs])

def weighted_stump(df, label_name, importance, verbose=0):
    dt=WDTree(df,label_name, importance)
    dt.train(1, verbose=verbose)
    return dt