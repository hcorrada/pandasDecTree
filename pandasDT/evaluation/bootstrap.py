import numpy as np

class Bootstrap(object):
    def __init__(self, size, n):
        self.n = n
        self.num = 0
        self.sample = None
        self.size = size

    def __iter__(self):
        return self

    def next(self):
        if self.num < self.n:
            self.sample = np.random.random_integers(0,self.size-1, self.size)
            self.num += 1
            return self.sample
        else:
            raise StopIteration()

    def get_out_of_sample(self):
        indxs = np.arange(self.size)
        sample = np.sort(self.sample)
        return filter(lambda x: np.sum((sample-x)==0)==0, indxs)


def bootestimate(df, label_name, classifier, evalfn, n=10):
    estimates = np.empty(n)
    nexamples = df.shape[0]
    bt = Bootstrap(nexamples, n)

    i=0
    for sample in bt:
        traindf = df.ix[sample,:]
        traindf.index = np.arange(nexamples)

        cl = classifier(traindf, label_name)

        out_of_sample = bt.get_out_of_sample()
        if len(out_of_sample)==0:
            estimates[i] = None
            i += 1
            continue

        testdf = df.ix[out_of_sample,:]
        testdf.index = np.arange(testdf.shape[0])

        preds = cl.predict(testdf)
        labels = testdf[label_name]
        estimates[i] = evalfn(preds, labels)
        i += 1
    return estimates
