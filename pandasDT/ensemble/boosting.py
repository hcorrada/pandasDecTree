import numpy as np
from pa03.dectree.WDTree import weighted_stump
from pa03.evaluation.eval import weighted_error_rate

class AdaBoostedStumps(object):
    def __init__(self, traindat, label_name):
        self.traindat = traindat
        self.label_name = label_name
        self.ensemble = []
        self.alpha = []

    def __repr__(self):
        out = 'AdaBoostedStumps with %d stumps\n' % len(self.ensemble)
        for i in xrange(len(self.ensemble)):
            out += '%d: %.3f\n' % (i, self.alpha[i])
            out += self.ensemble[i].__repr__() + '\n'
        return out

    def train(self, k=20):
        nexamples = self.traindat.shape[0]
        importance = 1/float(nexamples) * np.ones(nexamples)
        labels = self.traindat[self.label_name]

        for i in range(k):
            cl = weighted_stump(self.traindat, self.label_name, importance)
            self.ensemble.append(cl)
            preds = cl.predict(self.traindat)
            error = weighted_error_rate(preds, labels, importance)
            alpha = 0.5 * np.log((1-error) / error)
            self.alpha.append(alpha)
            importance = importance * np.exp(-alpha * preds * labels)
            importance = importance / float(np.sum(importance))

    def predict(self, testdat):
        nexamples = testdat.shape[0]
        nclass = len(self.ensemble)

        preds = np.empty((nexamples,nclass))
        for i in range(len(self.ensemble)):
            preds[:,i] = self.alpha[i] * self.ensemble[i].predict(testdat)
        return np.sign(np.sum(preds, axis=1))



