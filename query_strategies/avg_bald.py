import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import torch

class AvgBaldSampling(Strategy):
    def __init__(self, X, Y, Cens,  ids, net_args, random_seed=123):
        super(AvgBaldSampling, self).__init__(X, Y, Cens,  ids, net_args, random_seed)

    #updated for torch
    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        softplus = torch.nn.Softplus()
        stds = 1e-5 + softplus(samples[:,:,1])
        
        mean_stddev_all = stds.mean(1)
        entropy_expected = torch.log(mean_stddev_all)
        expected_entropy = torch.log(stds).mean(1)
        bald_cens = entropy_expected - expected_entropy

        stds = 1e-5 + softplus(samples[:,:,3])
        
        mean_stddev_all = stds.mean(1)
        entropy_expected = torch.log(mean_stddev_all)
        expected_entropy = torch.log(stds).mean(1)
        bald_unc = entropy_expected - expected_entropy

        mu_0  = samples[:,:,0]
        mu_1  = samples[:,:,2]
        pt = (mu_0 <= mu_1).float().mean(1)
        #t = torch.zeros_like(pt) # to make it use the mu_0
        t = (pt>0.1).float()
        return ((pt*torch.log(bald_unc))+((1-pt)*torch.log(bald_cens))).detach().numpy(), idxs_unlabeled
