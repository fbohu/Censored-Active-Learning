import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
import tensorflow as tf
from .causal_bald import mu_pi
from scipy import stats
import torch

class MuPiSampling(Strategy):
    def __init__(self, X, Y, Cens,  ids, net_args):
        super(MuPiSampling, self).__init__(X, Y, Cens,  ids, net_args)

    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        mu_0  = samples[:,:,0]
        mu_1  = samples[:,:,2]
        pt = (mu_0 >= mu_1).float().mean(1)
        t = torch.bernoulli(pt)
        scores =  mu_pi(mu_0, mu_1, t=t,  pt=pt, temperature=0.25)
        return scores.detach().numpy(), idxs_unlabeled