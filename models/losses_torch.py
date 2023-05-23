import torch

######################################
## File with loss Functions for PYTORCH
######################################
def combined_tobit(y, f):
    """ loss function for Tobit likelihood

    y_true: true regression labels
    yhat: estimated labels, in a TF distribution
    censored: a binary list where 1 is a censored observations
    """
    y_pred = f[:,0]
    softplus = torch.nn.Softplus()
    #torch.nn.functional.softplus
    #sigma = 1e-5 + torch.nn.functional.softplus(f[:,1])
    #sigma =  torch.exp(f[:,1])
    sigma =  1e-5 + softplus(f[:,1])
    y_true = y[:,0]#tf.cast(y[:,0], tf.float32)
    censored = y[:,1].int().bool() #tf.cast(y[:,1], tf.float32)
    norm = torch.distributions.Normal(loc=0., scale=1.)
    
    loglik_not_cens_arg = norm.log_prob((y_true-y_pred)/sigma).exp() / sigma
    loglik_cens_arg = 1. - norm.cdf((y_true-y_pred)/sigma)

    loglik_not_cens_arg = torch.clip(loglik_not_cens_arg, 0.0000001, 10000000)
    loglik_cens_arg = torch.clip(loglik_cens_arg, 0.0000001, 10000000)

    loglik =  torch.where(censored,  torch.log(loglik_cens_arg),  torch.log(loglik_not_cens_arg))
    
    negloglik = -1*(loglik.sum())
    return negloglik

def nll(y, f):
    """ loss function for Tobit likelihood

    y_true: true regression labels
    yhat: estimated labels, in a TF distribution
    censored: a binary list where 1 is a censored observations
    """
    y_pred = f[:,0]
    softplus = torch.nn.Softplus()
    #sigma = 1e-5 + softplus(f[:,1])
    #sigma = 1e-5 +torch.nn.functional.softplus(f[:,1])
    #sigma =  torch.exp(f[:,1])
    sigma =  1e-5 + softplus(f[:,1])
    y_true = y[:,0] #tf.cast(y[:,0], tf.float32)
    #y_pred = #tf.cast(y_pred, tf.float32)
    #sigma = #tf.cast(sigma, tf.float32)
    
    #norm = tfp.distributions.Normal(loc=0., scale=1.)
    norm = torch.distributions.Normal(loc=0., scale=1.)
    loglik_not_cens_arg = norm.log_prob((y_pred-y_true)/sigma).exp() / sigma
    loglik_not_cens_arg = torch.clip(loglik_not_cens_arg, 0.0000001, 10000000)
    negloglik = -1.0*(torch.log(loglik_not_cens_arg).sum())
    return negloglik


def combined_loss(y, f):
    loss = 0.0
    loss += combined_tobit(y, f[:,:2])
    if f.shape[-1] > 2:
        loss += nll(y, f[:,2:])
    return loss