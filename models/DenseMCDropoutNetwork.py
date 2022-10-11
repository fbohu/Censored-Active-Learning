import tensorflow as tf
import tensorflow_probability as tfp
from .models import Model
from .losses import tobit_nll, censored_multi_tilted_loss, combined_loss
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
tfd = tfp.distributions


class DenseMCDropoutNetwork(Model):
    """ A multilayer fully-connected Bayesian neural network with MonteCarlo (MC) dropout
    
    """
    def __init__(self, in_dims, hidden_dims,normalizer, dropout_p=0.50, lr_rate = 3e-4, ensemble_size=25):
        super(DenseMCDropoutNetwork, self).__init__(in_dims, hidden_dims, normalizer)

        ## Create Keras sequential model using the dims and MC dropout
        self.model = Sequential()
        # Input layer
        self.model.add(Input(shape=(in_dims,)))
        self.model.add(normalizer)

        for i in range(0,len(hidden_dims)):
            self.model.add(Dense(hidden_dims[i], activation='linear', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4)))
            self.model.add(tf.keras.layers.LeakyReLU())
            self.model.add(MCDropout(dropout_p))

        # Remove last MCDroput layer
        self.model.pop()
        # Output Layer     
        self.model.add(Dense(1+1+1+1))
        #self.model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
        #                                            scale=1e-5 + tf.math.softplus(t[...,1:]))))
        #self.model.add(Dense(out_dims))
        #print(self.model.summary())
        self.ensemble_size = ensemble_size

    def loss_func(self, y_true, yhat, censored):
        #return tobit_nll(y_true, yhat)
        return combined_loss(y_true, yhat)

    def sample(self, x):
        samples = np.zeros([self.ensemble_size, x.shape[0],4])
        for i in range(self.ensemble_size):
                samples[i] = self.model(x)

        return samples

    def predict(self, x, ensemble_size=500):
        samples = np.zeros([self.ensemble_size, x.shape[0]])
        for i in range(self.ensemble_size):
            samples[i] = self.model(x)[:,0]
        return np.mean(samples, axis=0)
        
    
class MCDropout(Dropout):
    """ A MonteCarlo (MC) dropout layer used for a MC dense network

    """
    def call(self, inputs):
        return super().call(inputs,training=True)