import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
from scipy import stats
import torch

class DuoUncertaintySampling(Strategy):
    def __init__(self,X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123):
        super(DuoUncertaintySampling, self).__init__(X, Y, Cens,  ids, net_args, x_val=x_val, y_val=y_val, random_seed=random_seed)

    #updated for torch
    def get_scores(self, n, plotting=False):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        mean_stddev_all_cens = samples[:,:,0].std(1)
        mean_stddev_all_obs = samples[:,:,1].std(1)
        if plotting:
            return torch.log(mean_stddev_all_obs+mean_stddev_all_cens).detach().numpy(), idxs_unlabeled,  torch.log(mean_stddev_all_cens), torch.log(mean_stddev_all_obs), 1.0

        return torch.log(mean_stddev_all_obs+mean_stddev_all_cens).detach().numpy(), idxs_unlabeled