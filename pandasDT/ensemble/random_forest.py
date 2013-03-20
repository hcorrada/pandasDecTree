from bagging import Bagger
from pa03.dectree.RTree import get_tree

def random_forest(df, label_name, ntrees=11, maxdepth=10, nfeatures_per_split=5):
    classifier = lambda df, label_name: get_tree(df, label_name, maxdepth=maxdepth, nfeatures_per_split=nfeatures_per_split)
    bag = Bagger(df, label_name, classifier)
    bag.train(ntrees)
    return bag
