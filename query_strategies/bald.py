import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
from scipy import stats
import torch

class BaldSampling(Strategy):
    def __init__(self,X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123):
        super(BaldSampling, self).__init__(X, Y, Cens,  ids, net_args, x_val=x_val, y_val=y_val, random_seed=random_seed)

    def get_scores(self, n, plotting=False):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        softplus = torch.nn.Softplus()
        stds = 1e-5 + softplus(samples[:,:,1])
        
        mean_stddev_all = stds.mean(1)
        entropy_expected = torch.log(mean_stddev_all)
        expected_entropy = torch.log(stds).mean(1)
        bald = entropy_expected - expected_entropy

        if plotting:
            return torch.log(bald).detach().numpy(), idxs_unlabeled, torch.ones_like(bald), torch.log(bald), torch.ones_like(mean_stddev_all)

        return bald.detach().numpy(), idxs_unlabeled
