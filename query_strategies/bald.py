import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
from scipy import stats
import torch

class BaldSampling(Strategy):
    def __init__(self, X, Y, Cens,  ids, net_args, random_seed=123):
        super(BaldSampling, self).__init__(X, Y, Cens,  ids, net_args, random_seed)

    #updated for torch
    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        softplus = torch.nn.Softplus()
        stds = 1e-5 + softplus(samples[:,:,1])
        
        mean_stddev_all = stds.mean(1)
        entropy_expected = torch.log(mean_stddev_all)
        expected_entropy = torch.log(stds).mean(1)
        bald = entropy_expected - expected_entropy

        return torch.log(bald).detach().numpy(), idxs_unlabeled

        '''  
    

    def get_scores(self, n):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        stds = 1e-5 + tf.math.softplus(samples[:,:,1])
        mean_stddev_all = tf.reduce_mean(stds, axis=0)
        entropy_expected = tf.math.log(mean_stddev_all)
        expected_entropy = tf.reduce_mean(tf.math.log(stds), axis=0)
        bald = entropy_expected - expected_entropy
        #stds = np.std(samples[:,:,0], axis = 0)
        #mean_stddev_all = np.mean(stds, axis = 0)
        #entropy_expected = np.log(stds)
        #expected_entropy = np.mean(np.log(stds), axis=0)
        #bald = entropy_expected - expected_entropy
        return bald.numpy(), idxs_unlabeled

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

        #return ids
