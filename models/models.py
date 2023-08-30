import tensorflow as tf
from tensorflow.keras import Sequential
from .losses import *
#from .losses_torch import *
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

class Model(tf.keras.Model):
    """ Base class for a model. Contains train functions and hyperparmeters
    """
    def __init__(self, in_dims, hidden_dims, normalizer, lr_rate = 0.003):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(lr=lr_rate)
        self.BATCH_SIZE = 32
        self.num_epochs = 1000
        self.quantiles = [0.05, 0.5, 0.95]
    
    def _train(self, x_data, y_data, censored):
        tmp = np.concatenate((y_data[:,np.newaxis], censored[:,np.newaxis]), axis=1)
        opt= tf.keras.optimizers.Adam(learning_rate=3e-3, clipnorm=1)
        self.compile(loss=lambda y, f: combined_loss(y, f),optimizer=opt)
        history = self.fit(x_data, tmp, epochs = 1000, verbose = 0,  shuffle=True, batch_size = 64)

    def call(self, x_data):
        ''' Forward pass of the model.
        '''
        return self.model(x_data)

    def predict(self, x_data):
        ''' Function that predicts the label on x_data.

        This function is dependend on the trained network. 
        Example: MC Dropout need multiple forward passes.

        '''
        raise NotImplementedError

    def evaluate(self, x_data, y_data):
        ''' Evaluate the trained network on the a given dataset.
        Evaluation will be based on NLL.
        '''
        tmp = np.concatenate((y_data[:,np.newaxis], y_data[:,np.newaxis]), axis=1) # quick hack to fit dimensions.
        preds = np.mean(self.sample(x_data), axis=0)
        #preds = np.mean(samples
        return nll(tmp, preds[:,0],preds[:,1])
        #return tf.reduce_mean((y_data-preds[:,0])**2)

    def loss_func(self, y_estimate, y_true):
        raise NotImplementedError