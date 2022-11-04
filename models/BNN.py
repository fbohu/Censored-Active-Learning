import torch
from torch.nn import Module
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from .losses_torch import *


class BayesianModule(Module):
    """A module that we can sample multiple times from given a single input batch.

    To be efficient, the module allows for a part of the forward pass to be deterministic.
    """
    k = None

    def __init__(self):
        super().__init__()

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, k: int):
        BayesianModule.k = k

        mc_input_BK = BayesianModule.mc_tensor(input_B, k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return mc_input_BK

    @staticmethod
    def unflatten_tensor(input: torch.Tensor, k: int):
        input = input.view([-1, k] + list(input.shape[1:]))
        return input

    @staticmethod
    def flatten_tensor(mc_input: torch.Tensor):
        return mc_input.flatten(0, 1)

    @staticmethod
    def mc_tensor(input: torch.tensor, k: int):
        mc_shape = [input.shape[0], k] + list(input.shape[1:])
        return input.unsqueeze(1).expand(mc_shape).flatten(0, 1)




class _ConsistentMCDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))

        self.p = p
        self.mask = None

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        self.mask = None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.reset_mask()

    def _get_sample_mask_shape(self, sample_shape):
        return sample_shape

    def _create_mask(self, input, k):
        mask_shape = [1, k] + list(self._get_sample_mask_shape(input.shape[1:]))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        return mask

    def forward(self, input: torch.Tensor):
        if self.p == 0.0:
            return input

        k = BayesianModule.k
        if self.training:
            # Create a new mask on each call and for each batch element.
            k = input.shape[0]
            mask = self._create_mask(input, k)
        else:
            if self.mask is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask = self._create_mask(input, k)

            mask = self.mask

        mc_input = BayesianModule.unflatten_tensor(input, k)
        mc_output = mc_input.masked_fill(mask, 0) / (1 - self.p)

        # Flatten MCDI, batch into one dimension again.
        return BayesianModule.flatten_tensor(mc_output)


# export


class ConsistentMCDropout(_ConsistentMCDropout):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call during training time.

    During eval time, a fixed mask is picked and kept until `reset_mask()` is called.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """
    pass


class ConsistentMCDropout2d(_ConsistentMCDropout):
    r"""Randomly zeroes whole channels of the input tensor.
    The channels to zero-out are randomized on every forward call.

    During eval time, a fixed mask is picked and kept until `reset_mask()` is called.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def _get_sample_mask_shape(self, sample_shape):
        return [sample_shape[0]] + [1] * (len(sample_shape) - 1)


class BayesianNN(BayesianModule):
    def __init__(self, in_dims, out_dims, hidden_dims, dropout_p=0.25, epochs = 500, lr_rate = 3e-4):
        super().__init__()

        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device= 'cpu'
        self.epochs = epochs
        self.lr = lr_rate
        layers = []
        for i, dims  in enumerate(hidden_dims):
            if i ==0:
                layers.append(nn.Linear(in_dims, dims))
                #layers.append(nn.LeakyReLU())
                layers.append(nn.GELU())
                layers.append(ConsistentMCDropout(p=dropout_p))
            else:
                layers.append(nn.Linear(dims, dims))
                layers.append(nn.GELU())
                #layers.append(nn.LeakyReLU())
                layers.append(ConsistentMCDropout(p=dropout_p))
                
        ## Remove last fully connected
        layers = layers[:-1]
        layers.append(nn.Linear(dims, out_dims))
        self.net = nn.Sequential(*layers)
            
        #self.fc1_drop = ConsistentMCDropout(p=dropout_p)
        #self.fc2 = nn.Linear(128, 128)
        #self.fc2_drop = ConsistentMCDropout(p=dropout_p)
        #self.fc2 = nn.Linear(128, 128)
        #self.fc3_drop = ConsistentMCDropout(p=dropout_p)
        #self.fc3 = nn.Linear(128, 128)
        #self.fc4_drop = ConsistentMCDropout(p=dropout_p)
        #self.fc4 = nn.Linear(128, 128)
        #self.fc5 = nn.Linear(128, 4)

    def mc_forward_impl(self, input: torch.Tensor):
        #input = F.leaky_relu(self.fc1_drop(self.fc1(input)))
        #input = F.leaky_relu(self.fc2_drop(self.fc2(input)))
        #input = F.leaky_relu(self.fc3_drop(self.fc3(input)))
        #input = self.fc5(input)
        output = self.net(input)

        return output


    def _train(self, x_data, y_data, censored):
        self.to(self.device)
        tmp = np.concatenate((y_data[:,np.newaxis], censored[:,np.newaxis]), axis=1)
        
        #optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        tmp = torch.tensor(tmp).float()
        #x_data = torch.tensor(x_data).float().clone().detach()
        dataset = torch.utils.data.TensorDataset(x_data, tmp)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.train()

        for i in range(0, self.epochs):
            for _, (data, target) in enumerate(train_dataloader):
                #data = data.to(device=self.device, non_blocking=True)
                #target = target.to(device=self.device, non_blocking=True)
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                optimizer.zero_grad()
                prediction = self(data, k=1).squeeze(1)
                #loss = combined_tobit(target, prediction)
                #loss = nll(target, prediction)
                loss = combined_loss(target, prediction)
                #loss = ((target-prediction)**2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

        self.cpu() 
        self.eval()
        #opt= tf.keras.optimizers.Adam(learning_rate=3e-3, clipnorm=1)

        #self.compile(loss=lambda y, f: combined_loss(y, f),optimizer=opt)
        #history = self.fit(x_data, tmp, epochs = 1000, verbose = 0,  shuffle=True, batch_size = 64)

    @torch.no_grad()
    def evaluate(self, x_data, y_data):
        ''' Evaluate the trained network on the a given dataset.
        Evaluation will be based on NLL.
        '''
        tmp = np.concatenate((y_data[:,np.newaxis], y_data[:,np.newaxis]), axis=1) # quick hack to fit dimensions.
        tmp = torch.tensor(tmp).float()
        preds = self.predict(x_data)
        return nll(tmp, preds[:,:2])

    @torch.no_grad()
    def sample(self, x, k= 20):
        self.to(self.device)
        self.eval()
        
        dataset = torch.utils.data.TensorDataset(x)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        N = x.shape[0]
        samples = torch.empty((N, k, 4))

        for i, (data, ) in enumerate(train_dataloader):

            lower = i * train_dataloader.batch_size
            upper = min(lower + train_dataloader.batch_size, N)
            #samples[lower:upper].copy_(self(data.to(self.device, non_blocking=True), k))#, non_blocking=True)
            samples[lower:upper].copy_(self(data.to(self.device), k))#, non_blocking=True)
        
        self.cpu()
        return samples.cpu().detach()
    
    @torch.no_grad()
    def predict(self, x, k=20):
        self.to(self.device)
        self.eval()
        dataset = torch.utils.data.TensorDataset(x)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        N = x.shape[0]
        samples = torch.empty((N, k, 4))

        for i, (data, ) in enumerate(train_dataloader):

            lower = i * train_dataloader.batch_size
            upper = min(lower + train_dataloader.batch_size, N)
            #samples[lower:upper].copy_(self(data.to(self.device, non_blocking=True), k))#, non_blocking=True)
            samples[lower:upper].copy_(self(data.to(self.device), k))#, non_blocking=True)
        #samples = self(x, k=20)
        self.cpu()
        return samples.cpu().detach().mean(1)
        


class BayesianConvNN(BayesianModule):
    def __init__(self, in_dims, out_dims, hidden_dims, dropout_p=0.25, epochs = 500, lr_rate = 3e-4):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'
        if self.device == 'cuda': print("Cuda Available. Running on GPU.")
        self.epochs = epochs
        self.lr = lr_rate
        layers = []
        layers.append(nn.Conv2d(1, 32, kernel_size=5))
        layers.append(ConsistentMCDropout2d(p=dropout_p))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(32, 64, kernel_size=5))
        layers.append(ConsistentMCDropout2d(p=dropout_p))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.GELU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1024, 128))
        layers.append(nn.GELU())
        layers.append(ConsistentMCDropout(p=dropout_p))
        layers.append(nn.Linear(128, 4))
        
        self.net = nn.Sequential(*layers)

    def mc_forward_impl(self, input: torch.Tensor):
        output = self.net(input)
        return output

    def _train(self, x_data, y_data, censored):
        self.to(self.device)
        tmp = np.concatenate((y_data[:,np.newaxis], censored[:,np.newaxis]), axis=1)
        
        #optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        tmp = torch.tensor(tmp).float()
        #x_data = torch.tensor(x_data).float().clone().detach()
        dataset = torch.utils.data.TensorDataset(x_data, tmp)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.train()

        for i in range(0, self.epochs):
            for _, (data, target) in enumerate(train_dataloader):
                data = data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                #data = data.to(device=self.device)
                #target = target.to(device=self.device)
                optimizer.zero_grad()
                prediction = self(data, k=1).squeeze(1)
                #loss = combined_tobit(target, prediction)
                #loss = nll(target, prediction)
                loss = combined_loss(target, prediction)
                #loss = ((target-prediction)**2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

        self.cpu() 
        self.eval()
        #opt= tf.keras.optimizers.Adam(learning_rate=3e-3, clipnorm=1)

        #self.compile(loss=lambda y, f: combined_loss(y, f),optimizer=opt)
        #history = self.fit(x_data, tmp, epochs = 1000, verbose = 0,  shuffle=True, batch_size = 64)

    @torch.no_grad()
    def evaluate(self, x_data, y_data):
        ''' Evaluate the trained network on the a given dataset.
        Evaluation will be based on NLL.
        '''
        tmp = np.concatenate((y_data[:,np.newaxis], y_data[:,np.newaxis]), axis=1) # quick hack to fit dimensions.
        tmp = torch.tensor(tmp).float()
        preds = self.predict(x_data)
        return nll(tmp, preds[:,:2])

    @torch.no_grad()
    def sample(self, x, k= 20):
        self.to(self.device)
        self.eval()
        
        dataset = torch.utils.data.TensorDataset(x)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        N = x.shape[0]
        samples = torch.empty((N, k, 4))

        for i, (data, ) in enumerate(train_dataloader):

            lower = i * train_dataloader.batch_size
            upper = min(lower + train_dataloader.batch_size, N)
            samples[lower:upper].copy_(self(data.to(self.device, non_blocking=True), k))#, non_blocking=True)
            #samples[lower:upper].copy_(self(data.to(self.device), k))#, non_blocking=True)
        
        self.cpu()
        return samples.cpu().detach()
    
    @torch.no_grad()
    def predict(self, x, k=20):
        self.to(self.device)
        self.eval()
        dataset = torch.utils.data.TensorDataset(x)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        N = x.shape[0]
        samples = torch.empty((N, k, 4))

        for i, (data, ) in enumerate(train_dataloader):

            lower = i * train_dataloader.batch_size
            upper = min(lower + train_dataloader.batch_size, N)
            samples[lower:upper].copy_(self(data.to(self.device, non_blocking=True), k))#, non_blocking=True)
            #samples[lower:upper].copy_(self(data.to(self.device), k))#, non_blocking=True)
        #samples = self(x, k=20)
        self.cpu()
        return samples.cpu().detach().mean(1)
        