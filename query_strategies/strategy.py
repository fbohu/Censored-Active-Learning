import numpy as np
from models import BNN
import scipy.stats
import torch

class Strategy:
    def __init__(self, X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123, dropout_p=0.50, beta = 1.0):
        self.X = X
        self.Y = Y
        self.X_val = x_val
        self.Y_val = y_val
        self.Cens = Cens
        self.ids = ids
        self.name = type(self).__name__ + net_args['dataset']+net_args['size']
        self.net_args = net_args
        if self.net_args['in_features'] < 0:
            print("running mnist")
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
        scores = np.exp(scores) 
        scores[np.isnan(scores)] = 1e-7 
        p = scores/np.sum(scores) 
        idx = np.random.choice(  
                        idxs_unlabeled, replace=False, p=p, size=n,
                    )
        return idx


    def update(self, new_ids, reset_net = True):
        self.ids = new_ids
        # Test if the network should be reset
        if reset_net:
            self.net = self.create_model()

    def train(self):
        self.net._train(self.X[self.ids], self.Y[self.ids], self.Cens[self.ids], self.X_val, self.Y_val)

    def evaluate(self, x_test, y_test, censoring_test):
        nll = self.net.evaluate(x_test, y_test).detach().numpy()
        preds = self.net.predict(x_test).detach()[:,0]
        tobitnll = self.net.evaluate_tobit(x_test, y_test, censoring_test).detach().numpy()
        y_test = torch.exp(y_test)
        preds = torch.exp(preds)
        #print(preds.shape)
        #print(y_test.shape)
        #print(censoring_test.shape)
        #print(censoring_test)
        #print(torch.maximum(y_test-preds, torch.zeros_like(preds)))
        #print(torch.maximum(y_test-preds, torch.zeros_like(preds)))
        #print(torch.mean(torch.maximum(y_test-preds, torch.zeros_like(preds))))
        #print(torch.where(torch.tensor(censoring_test), torch.maximum(y_test-preds, torch.zeros_like(preds)), abs(y_test-preds)))
        #print(torch.where(torch.tensor(censoring_test) == 1.0, torch.maximum(y_test-preds, torch.zeros_like(preds)), abs(y_test-preds)))
        #print(torch.mean(torch.where(torch.tensor(censoring_test) == 1.0, torch.maximum(y_test-preds, torch.zeros_like(preds)), abs(y_test-preds))))
        mae = torch.mean(torch.abs(y_test-preds)[censoring_test == 0.0]).detach().numpy()
        mae_hinge = torch.mean(torch.maximum(y_test-preds, torch.zeros_like(preds))[censoring_test == 1.0])
        mae_hinge = torch.nan_to_num(mae_hinge, nan=0.0).detach().numpy() # for the case with true labels.
        #mae_hinge = torch.mean(
        #                torch.where(
        #                    torch.tensor(censoring_test) == 1.0, torch.maximum(y_test-preds, torch.zeros_like(preds)), abs(y_test-preds)
        #                    )
        #            )

        #mae_hinge = np.mean(np.where(censoring_test == 1, np.max(y_test-preds), 0))

        return nll, tobitnll, mae, mae_hinge, mae+mae_hinge 

    def create_model(self):
        ''' Functions that creates a network based on network arguments. 

        This functions allows for resetting of the network after each query.
        '''
        if self.net_args['in_features'] < 0:
            return BNN.BayesianConvNN(self.net_args['in_features'],
                            self.net_args['out_features'],
                            self.net_args['hidden_size'],
                            dropout_p=self.net_args['dropout_p'],
                            epochs = self.net_args['epochs'],
                            lr_rate =self.net_args['lr_rate'],
                            name=self.name)
        else:
            return BNN.BayesianNN(self.net_args['in_features'],
                            self.net_args['out_features'],
                            self.net_args['hidden_size'],
                            dropout_p=self.net_args['dropout_p'],
                            epochs = self.net_args['epochs'],
                            lr_rate =self.net_args['lr_rate'],
                            name=self.name)


    def predict(self, x):
        return self.net.predict(x)

    def get_embedding(self):
        pass
