import numpy as np
import torch
from .strategy import Strategy

class RandomSampling(Strategy):
    #def __init__(self, X, Y, Cens,  ids, net_args, random_seed=123):
    def __init__(self,X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123):
        super(RandomSampling, self).__init__(X, Y, Cens,  ids, net_args, x_val=x_val, y_val=y_val, random_seed=random_seed)

    #def query(self, n):
    #    return np.random.choice(np.where(self.ids==0)[0], n, replace=False)

    def get_scores(self, n, plotting=False):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]

        if plotting:
            return np.ones_like(idxs_unlabeled), idxs_unlabeled, torch.ones_like(torch.tensor(idxs_unlabeled)), torch.ones_like(torch.tensor(idxs_unlabeled))

        return np.ones_like(idxs_unlabeled), idxs_unlabeled


    #def train(self):
    #    pass
