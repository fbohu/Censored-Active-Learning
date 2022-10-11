import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, X, Y, Cens,  ids, net_args, random_seed=123):
        super(RandomSampling, self).__init__(X, Y, Cens,  ids, net_args, random_seed)

    #def query(self, n):
    #    return np.random.choice(np.where(self.ids==0)[0], n, replace=False)

    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        return np.ones_like(idxs_unlabeled), idxs_unlabeled


    #def train(self):
    #    pass
