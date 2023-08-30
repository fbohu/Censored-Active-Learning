import numpy as np
from .strategy import Strategy
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn.functional as F

class CensBaldSampling(Strategy):
    def __init__(self,X, Y, Cens,  ids, net_args, x_val, y_val, random_seed = 123):
        super(CensBaldSampling, self).__init__(X, Y, Cens,  ids, net_args, x_val=x_val, y_val=y_val, random_seed=random_seed)

    def get_scores(self, n, plotting=False):
        idxs_unlabeled = np.arange(self.Y.shape[0])[~self.ids]
        samples = self.net.sample(self.X[idxs_unlabeled])
        
        softplus = torch.nn.Softplus()
        stds = 1e-5 + softplus(samples[:,:,1])
        

        tobit_dist = torch.distributions.normal.Normal(loc=samples[:,:,0], scale=stds)
        logits = F.sigmoid(samples[:,:,-1])+1e-7
        pdf_part = tobit_dist.log_prob(samples[:,:,2])
        cdf_part = torch.log(1.0-tobit_dist.cdf(samples[:,:,2]))
        predictive_entropy = ((1-logits)*pdf_part + logits*cdf_part).mean(1)
        
        tobit_dist = torch.distributions.normal.Normal(loc=samples[:,:,0], scale=stds)
        logits = F.sigmoid(samples[:,:,-1]).mean(1).unsqueeze(1) + 1e-7
        # 1 if censored
        pdf_part = tobit_dist.log_prob(samples[:,:,2]).mean(1).unsqueeze(1)
        cdf_part = torch.log(1.0-tobit_dist.cdf(samples[:,:,2]).mean(1)).unsqueeze(1)
        conditional_entropy  = ((1-logits)*pdf_part + logits*cdf_part).squeeze(1)

        bald = (predictive_entropy - conditional_entropy)

        # plus entropy for classification
        # bald form here https://github.com/baal-org/baal/blob/master/baal/active/heuristics/heuristics.py
        probs = torch.unsqueeze(F.sigmoid(samples[:,:,-1]), 1) +1e-7

        entropy_expected_p = (-probs*torch.log(probs)).mean(2).sum(1)

        pb = probs.mean(2)
        expected_entropy_p = (-pb*torch.log(pb)).sum(1)

        uncertainties = expected_entropy_p - entropy_expected_p

        bald = (bald + uncertainties).clamp(min=1e-07, max=5.0)

        if plotting:
            return bald.detach().numpy(), idxs_unlabeled, bald.unsqueeze(1).detach().numpy(), uncertainties.unsqueeze(1).detach().numpy(), logits

        return bald.detach().numpy(), idxs_unlabeled
        