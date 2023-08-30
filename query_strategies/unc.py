import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
from scipy import stats
import torch

class UncertaintySampling(Strategy):
    def __init__(self,X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123):
        super(UncertaintySampling, self).__init__(X, Y, Cens,  ids, net_args, x_val=x_val, y_val=y_val, random_seed=random_seed)

    #updated for torch
    def get_scores(self, n, plotting=False):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        mean_stddev_all = samples[:,:,0].std(1)
        if plotting:
            return torch.log(mean_stddev_all).detach().numpy(), idxs_unlabeled,  torch.ones_like(torch.tensor(idxs_unlabeled)), torch.log(mean_stddev_all), torch.ones_like(mean_stddev_all)

        return mean_stddev_all.detach().numpy(), idxs_unlabeled