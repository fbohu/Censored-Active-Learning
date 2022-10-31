import numpy as np
from models import BNN
import scipy.stats
import torch

class Strategy:
    def __init__(self, X, Y, Cens,  ids, net_args, random_seed = 123, dropout_p=0.50, beta = 1.0):
        self.X = X
        self.Y = Y
        self.Cens = Cens
        self.ids = ids
        self.net_args = net_args
        self.beta = 0.25
        torch.manual_seed(123)
        self.dropout_p = dropout_p
        self.net = self.create_model()

        self.n_pool = Y.shape[0]

    def get_scores(self, n):
        pass

    def query(self, n):
        # Importance weighted sampling.
        scores, idxs_unlabeled = self.get_scores(n)
        scores = np.exp(scores) # used to get good results
        scores[np.isnan(scores)] = 1e-7 # used to get good results
        p = scores/np.sum(scores) # used to get good results
        idx = np.random.choice(  # used to get good results
                        idxs_unlabeled, replace=False, p=p, size=n,
                    )
        
        #p = scores = + scipy.stats.gumbel_r.rvs(
        #                loc=0, scale=1, size=len(scores), random_state=None,
        #            )

        #return idx
        #ids_ = p.argsort()[-n:][::-1]
        return idxs_unlabeled[ids_]


    def update(self, new_ids, reset_net = True):
        self.ids = new_ids
        # Test if the network should be reset
        if reset_net:
            self.net = self.create_model()

    def train(self):
        self.net._train(self.X[self.ids], self.Y[self.ids], self.Cens[self.ids])

    def evaluate(self, x_test, y_test):
        return self.net.evaluate(x_test, y_test).detach().numpy()

    def create_model(self):
        ''' Functions that creates a network based on network arguments. 

        This functions allows for resetting of the network after each query.
        '''
        # Clear Keras backend
    
        return BNN.BayesianNN(self.net_args['in_features'],
                            self.net_args['out_features'],
                            self.net_args['hidden_size'],
                            dropout_p=self.net_args['dropout_p'],
                            epochs = self.net_args['epochs'],
                            lr_rate =self.net_args['lr_rate'])
        #return DenseMCDropoutNetwork.DenseMCDropoutNetwork(self.net_args['in_features'],
        #                                                self.net_args['hidden_size'],
         #                                               input_normalizer)

    def predict(self, x):
        return self.net.predict(x)

    def get_embedding(self):
        pass
