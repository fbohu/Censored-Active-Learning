import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

######################################
## File with loss funcitons, include Tobit and CQNN
######################################
def tobit_nll(yobs, yhat):
    """ loss function for Tobit likelihood

    y_true: true regression labels
    yhat: estimated labels, in a TF distribution
    censored: a binary list where 1 is a censored observations
    """
    y_pred = yhat.mean()
    sigma = yhat.stddev()
    #sigma = 1.0
    y_true = tf.cast(tf.expand_dims(yobs[:,0],1), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    censored = tf.cast(tf.expand_dims(yobs[:,1],1), tf.float32)

    norm = tfp.distributions.Normal(loc=0., scale=1.)

    loglik_not_cens_arg = norm.prob((y_true-y_pred)/sigma) / sigma
    loglik_cens_arg = 1 - norm.cdf((y_true-y_pred)/sigma)

    loglik_not_cens_arg = tf.clip_by_value(loglik_not_cens_arg, 0.0000001, 10000000)
    loglik_cens_arg = tf.clip_by_value(loglik_cens_arg, 0.0000001, 10000000)

    loglik = tf.multiply(tf.math.log(loglik_not_cens_arg),(1-censored))+ tf.multiply(tf.math.log(loglik_cens_arg),(censored))
    #loglik = tf.where(censored, tf.math.log(loglik_cens_arg),  tf.math.log(loglik_not_cens_arg))
    negloglik = -1.0*tf.reduce_sum(loglik)
    return negloglik

def create_multi_output_target(y, quantiles):
    """ Function to create desired outputs for CQNN
    """
    y_ = y[:,np.newaxis]
    for _ in range(len(quantiles)-1):
        y_ = np.concatenate((y_, y[:,np.newaxis]), axis=1)
    return y_


def tilted_loss(theta, error):
    """ Tilted loss funcion
    """
    return K.maximum(theta*error, (theta-1)*error)

def censored_multi_tilted_loss(quantiles, y, f):
    """ loss function for CQNN
    
    y: true regression labels, in a [n, len(quantiles)+1]
    yf: estimated quantiles [n, len(quantiles)+1]
    quantiles: a list of estimated quantiles.
    """
    loss = 0.0
    y = tf.cast(y, tf.float32)
    f = tf.cast(f, tf.float32)
    
    treshold_values = tf.where(y[:,-1] == -1.0, y[:,0], -np.inf)

    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k] - K.maximum(treshold_values, f[:,k]))
        loss += tilted_loss(theta=q, error=e)
    return loss



def combined_tobit(y, f, std):
    """ loss function for Tobit likelihood

    y_true: true regression labels
    yhat: estimated labels, in a TF distribution
    censored: a binary list where 1 is a censored observations
    """
    y_pred = f#.mean()
    #sigma = tf.ones_like(f)*0.5
    sigma = 1e-5 + tf.math.softplus(std)
    #sigma = 1.0
    y_true = tf.cast(y[:,0], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    censored = tf.cast(y[:,1], tf.float32)

    norm = tfp.distributions.Normal(loc=0., scale=1.)

    loglik_not_cens_arg = norm.prob((y_true-y_pred)/sigma) / sigma
    loglik_cens_arg = 1 - norm.cdf((y_true-y_pred)/sigma)

    loglik_not_cens_arg = tf.clip_by_value(loglik_not_cens_arg, 0.0000001, 10000000)
    loglik_cens_arg = tf.clip_by_value(loglik_cens_arg, 0.0000001, 10000000)

    loglik = tf.multiply(tf.math.log(loglik_not_cens_arg),(1-censored))+ tf.multiply(tf.math.log(loglik_cens_arg),(censored))
    #loglik = tf.where(censored, tf.math.log(loglik_cens_arg),  tf.math.log(loglik_not_cens_arg))
    negloglik = -1.0*tf.reduce_sum(loglik)
    return negloglik


def nll(y, f, std):
    """ loss function for Tobit likelihood

    y_true: true regression labels
    yhat: estimated labels, in a TF distribution
    censored: a binary list where 1 is a censored observations
    """
    y_pred = f
    sigma = 1e-5 + tf.math.softplus(std)
    y_true = tf.cast(y[:,0], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    
    norm = tfp.distributions.Normal(loc=0., scale=1.)
    loglik_not_cens_arg = norm.prob((y_true-y_pred)/sigma) / sigma
    loglik_not_cens_arg = tf.clip_by_value(loglik_not_cens_arg, 0.0000001, 10000000)
    negloglik = -1.0*tf.reduce_sum(tf.math.log(loglik_not_cens_arg))
    return negloglik



def combined_loss(y, f):
    loss = 0.0
    loss += combined_tobit(y, f[:,0], f[:,1])
    #loss += tf.reduce_mean((y[:,0] - f[:,-1])**2)
    loss += nll(y, f[:,2],f[:,3])
    #y = tf.cast(y, tf.float32)
    # = tf.cast(f, tf.float32)
    return loss