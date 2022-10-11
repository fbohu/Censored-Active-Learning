from .models import Model
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input

class BayesianDensityNetwork(Model):
    def __init__(self, in_dims, out_dims, hidden_dims, lr_rate = 3e-4, reparam=True):
        super(BayesianDensityNetwork, self).__init__(in_dims, out_dims, hidden_dims)

        ## Create Keras sequential model using the dims and MC dropout
        self.model = Sequential()
        # Input layer
        self.model.add(Input(shape=(in_dims,)))
        for i in range(0,len(hidden_dims)):
            self.model.add(tfp.layers.DenseReparameterization(hidden_dims[i], activation=tf.nn.relu))
        # Output Layer     
        self.model.add(tfp.layers.DenseReparameterization(out_dims))

        # set train_size
        self.train_size = 1

    def loss_func(self, y_estimate, y_true):
        # TODO: Find way to scale the KL divergence by size of train_set
        return tf.reduce_mean((y_estimate-y_true)**2)+1/self.train_size*sum(self.model.losses)
    
    def train(self, x_data, y_data):
        # Set the train_size to scale the KL divergence
        self.train_size = x_data.shape[0]

        data_train = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(10).batch(self.BATCH_SIZE)
        for epoch in range(0, self.num_epochs):
            for batch, (x, y) in enumerate(data_train):
                self.train_step(x, y)
                #print('\rEpoch [%d/%d] Batch: %d' %(epoch+1, self.num_epochs, batch), end='')

    def predict(self, x, ensemble_size=5):
        preds = np.zeros([ensemble_size, x.shape[0],1])
        for i in range(ensemble_size):
            preds[i] = self.model(x)

        return np.mean(preds, axis=0)
        