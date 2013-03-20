from DTree import DTree
from random_scoring import best_split


class RTree(DTree):
    def __init__(self, traindat, label_name, nfeatures_per_split):
        super(RTree,self).__init__(traindat, label_name)
        self.nfeatures_per_split=nfeatures_per_split

    def split_func(self, indxs, features_to_use):
        return best_split(self.traindat.ix[indxs,:], features_to_use, self.label_name, self.nfeatures_per_split)

def get_tree(df, label_name, maxdepth=10, nfeatures_per_split = 5, verbose=0):
    # make tuning set
    dt=RTree(df,label_name, nfeatures_per_split)
    dt.train(maxdepth, verbose=verbose)
    return dt
