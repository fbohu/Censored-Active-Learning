import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
import tensorflow as tf
from .causal_bald import mu_tau
from scipy import stats
import torch

class MuTauSampling(Strategy):
    def __init__(self, X, Y, Cens,  ids, net_args):
        super(MuTauSampling, self).__init__(X, Y, Cens,  ids, net_args)

    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled]).detach()
        mu_0  = samples[:,:,0]
        mu_1  = samples[:,:,2]
        pt = (mu_0 >= mu_1).float().mean(1)
        t = torch.bernoulli(pt)
        scores =  mu_tau(mu_0, mu_1, t=t,  pt=pt, temperature=0.25)
        return scores.detach().numpy(), idxs_unlabeled
        '''  

        sorted_index_array = np.argsort(bald[:,0])
        print("len " + str(len(sorted_index_array[-n:])))
        #print(sorted_index_array)
        #print(idxs_unlabeled)
        print(idxs_unlabeled[sorted_index_array[-n:]])
        print(idxs_unlabeled[sorted_index_array[-n:]])
        print(self.X.numpy()[idxs_unlabeled[sorted_index_array[-n:]]])
        max_ = np.nanargmax(bald)
        plt.plot(self.X.numpy()[idxs_unlabeled], bald)
        plt.show()
        return idxs_unlabeled[sorted_index_array[-n:]]
        ids.append(idxs_unlabeled[max_])


        init_bald = np.argmax(np.log(np.mean(sigma1_, axis=0))-(np.mean(np.log(sigma1_), axis=0)))
        ids = [init_bald]
        sel = sigma1_[:,ids]

        for _ in range(1, n):
            covariance = []
            entropy = []
            for i in range(0, sigma1_.shape[-1]):
                if i in ids:
                    covariance.append(np.nan)
                    entropy.append(np.nan)
                else:        
                    covariance.append(0.5*np.log(np.linalg.det(np.cov(np.concatenate((sel, sigma1_[:,i][:,np.newaxis]), axis=1).T))) - _*np.mean(np.log(np.concatenate((sel, sigma1_[:,i][:,np.newaxis]), axis=1))))
                    entropy.append(_*np.var(np.log(np.concatenate((sel, sigma1_[:,i][:,np.newaxis]), axis=1))))
            
            plt.plot(self.X.numpy()[idxs_unlabeled], entropy)
            plt.show()
            plt.plot(self.X.numpy()[idxs_unlabeled], covariance)
            plt.show()
            max_ = np.nanargmax(covariance)
            ids.append(idxs_unlabeled[max_])
            sel = np.concatenate((sel, sigma1_[:,max_][:,np.newaxis]), axis=1)
        '''

        return ids
