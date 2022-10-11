import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from scipy import stats

class CensBaldSampling(Strategy):
    def __init__(self, X, Y, Cens,  ids, net_args, random_seed=123):
        super(CensBaldSampling, self).__init__(X, Y, Cens,  ids, net_args, random_seed)

    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])

        softplus = torch.nn.Softplus()
        stds_censored = 1e-5 + softplus(samples[:,:,1])
        stds_uncensored = 1e-5 + softplus(samples[:,:,3])
        stds = stds_censored+stds_uncensored
        
        mean_stddev_all = stds.mean(1)
        entropy_expected = torch.log(mean_stddev_all)
        expected_entropy = torch.log(stds).mean(1)
        bald = entropy_expected - expected_entropy

        return torch.log(bald).detach().numpy(), idxs_unlabeled
        