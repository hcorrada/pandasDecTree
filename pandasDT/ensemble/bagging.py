from pa03.evaluation.bootstrap import Bootstrap
import numpy as np

class Bagger(object):
    def __init__(self, traindat, label_name, classifier):
        self.traindat = traindat
        self.label_name = label_name
        self.classifier = classifier
        self.ensemble = []

    def __repr__(self):
        out = 'Bagger with %d classifiers' % len(self.ensemble)
        return out

    def train(self, n=11):
        nexamples = self.traindat.shape[0]
        bt = Bootstrap(nexamples, n)

        for sample in bt:
            traindf = self.traindat.ix[sample,:]
            traindf.index = np.arange(nexamples)

            self.ensemble.append(self.classifier(traindf, self.label_name))

    def predict(self, testdat):
        nexamples = testdat.shape[0]
        nclass = len(self.ensemble)

        preds = np.empty((nexamples, nclass))
        for i in range(nclass):
            preds[:,i] = self.ensemble[i].predict(testdat)

        print preds
        return np.sign(np.sum(preds, axis=1))




