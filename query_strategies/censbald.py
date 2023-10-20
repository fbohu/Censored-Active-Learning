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
        logits = F.sigmoid(samples[:,:,-1])+1e-7
        
                # Compute first time in the mutual information
        ## First compute the entropy of H(y|l,x)
        tobit_dist = torch.distributions.normal.Normal(loc=samples[:,:,0].mean(1), scale=stds.mean(1))
        pdf_part = tobit_dist.log_prob(samples[:,:,2].mean(1))
        cdf_part = torch.log(1.0-tobit_dist.cdf(samples[:,:,2].mean(1)))
        predictive_entropy =((1-logits.mean(1))*pdf_part + logits.mean(1)*cdf_part)

        ## entropy of H(l|x)
        probs = torch.unsqueeze(F.sigmoid(samples[:,:,4]), 1) +1e-7
        pb = probs.mean(2)
        expected_entropy_p = (-pb*torch.log(pb)).sum(1)
        # total entropy for first term
        expected_entropy = expected_entropy_p+predictive_entropy

        # Compute second term in the mutual information (H[y|l, x, theta])
        tobit_dist = torch.distributions.normal.Normal(loc=samples[:,:,0], scale=stds)
        logits = F.sigmoid(samples[:,:,-1]).unsqueeze(1) + 1e-7

        # entropy of H(l| x,theta)
        pdf_part = tobit_dist.log_prob(samples[:,:,2]).unsqueeze(1)
        cdf_part = torch.log(1.0-tobit_dist.cdf(samples[:,:,2])).unsqueeze(1)
        conditional_entropy_p  = ((1-logits)*pdf_part + logits*cdf_part).squeeze(1).mean(1)
        entropy_expected_p = (-probs*torch.log(probs)).mean(2).sum(1)

        likelihoods_contribution = predictive_entropy-conditional_entropy_p
        classes_contribution = expected_entropy_p-entropy_expected_p
        
        predictive_entropy = predictive_entropy+expected_entropy_p
        conditional_entropy = conditional_entropy_p+entropy_expected_p

        bald = (predictive_entropy - conditional_entropy).clamp(min=1e-07, max=5.0)

        if plotting:
            return bald.detach().numpy(), idxs_unlabeled, likelihoods_contribution.unsqueeze(1).detach().numpy(), classes_contribution.unsqueeze(1).detach().numpy(), torch.zeros_like(classes_contribution.unsqueeze(1)).detach().numpy()

        return bald.detach().numpy(), idxs_unlabeled
        