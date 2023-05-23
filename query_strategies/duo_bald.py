import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
from scipy import stats
import torch

class DuoBaldSampling(Strategy):
    def __init__(self,X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123):
        super(DuoBaldSampling, self).__init__(X, Y, Cens,  ids, net_args, x_val=x_val, y_val=y_val, random_seed=random_seed)

    def get_scores(self, n, plotting=False):
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

        if plotting:
            return torch.log(bald_unc).detach().numpy()+torch.log(bald_cens).detach().numpy(), idxs_unlabeled, torch.log(bald_cens), torch.log(bald_unc)

        return torch.log(bald_unc).detach().numpy()+torch.log(bald_cens).detach().numpy(), idxs_unlabeled
