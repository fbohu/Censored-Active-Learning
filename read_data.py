import pandas as pd
import numpy as np
import copy
import h5py
import torch
from data.synth import *
from sklearn.datasets import make_regression, make_friedman1

def get_dataset(name):
    if name == 'synth':
        return get_synth()
    elif name == 'ds1':
        return get_ds1()
    elif name == 'ds2':
        return get_ds1()
    elif name == 'ds3':
        return get_ds3()
    elif name == 'gsbg':
        return get_gsbg()
    elif name == 'support':
        return get_support()
    elif name == 'IHC4':
        return get_IHC4()
    elif name == 'sim':
        return get_sim()
    elif name == 'whas':
        return get_whas()
    elif name == 'sklearn':
        return get_sklearn()
    elif name == 'cbald':
        return get_causalbad()
        


def split_data(x, y_cens, y_true, censoring,test_size = 100, verbose = False):
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)
    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    print(x.shape[0])
    print(test_ids.shape)
    y_train = y_cens[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y_true[np.isin(np.arange(x.shape[0]), test_ids)]
    if verbose:
        print(x_train.shape)
        print(y_train.shape)
        print(censoring_train.shape)
        print(x_test.shape)
        print(y_test.shape)
    return x_train, y_train, censoring_train, x_test, y_test

def get_synth():
    """Data is generated as follows:
    - Define latent function f(x) = 2 + 0.5*sin(2x) + x/10
    - Generetae observations from the latent function y_obs (assuming some small observation noise, let's focus on censoring)
    - Select the points in the oscillation peaks. 
    - Apply a p_c% manual censoring to those points sampled uniformly between [0.2, 0.3] for all selected points.
    """

    np.random.seed(10)
    n = 10000+1000
    # Define underlying function
    x = np.linspace(2, 6, n)
    y_true = 0.5*np.sin(2*x) + 2 #+ x/10
    #y_true = 0.5*x + 2

    # Generate noisy observations 
    y_obs = y_true + np.random.normal(loc=0, scale=0.01*x, size=x.shape[0]) ## Heterogenue noise
    #y_obs = y_true + np.random.normal(loc=0, scale=0.01, size=x.shape[0]) ## Homo noise
    y_cens = copy.deepcopy(y_obs)

    cens_levl = 2.2
    censoring = np.int32(0.5*np.sin(2*x) + 2 >cens_levl) 
    #censoring = np.random.choice(2, n, p=[0.1, 0.9])*censoring # this can be used to uncensor some.
    p_c = np.random.uniform(low=0.10, high=0.30, size=np.sum(censoring==1))
    #y_cens[censoring == 1] = y_obs[censoring == 1]*(1-p_c)
    y_cens[censoring == 1] = cens_levl + np.random.normal(loc=0, scale=0.01, size=sum(censoring))

    x = x.reshape(n,1)
    return split_data(x, y_cens, y_true, censoring, test_size = 1000, verbose = True)
    
def get_ds1():
    ds1 = make_ds1(True, 10000+1000, 1)
    cens = (-1*ds1['y_star'] > -1*ds1['y'])+0
    return split_data(ds1['X'].T, -1*ds1['y'], -1*ds1['y_star'],cens,test_size=1000, verbose = True)

def get_ds2():
    ds2 = make_ds2(True,10000+1000, 1)
    cens = (-1*ds2['y_star'] > -1*ds2['y'])+0
    return split_data(ds2['X'].T, -1*ds2['y'], -1*ds2['y_star'],cens,test_size=1000, verbose = True)

def get_ds3():
    ds3 = make_ds3(True, 10000+1000, 1)
    cens = (-1*ds3['y_star'] > -1*ds3['y'])+0
    return split_data(ds3['X'].T, -1*ds3['y'], -1*ds3['y_star'],cens,test_size=1000, verbose = True)

def get_gsbg():
    f1 = h5py.File("data/gbsg_cancer_train_test.h5",'r+')   
    x_train = f1['train']['x']
    y_orig = f1['train']['t']
    censoring_train = f1['train']['e']
    x_test = f1['test']['x']
    y_test = f1['test']['t']
    y_train = y_orig/max(y_orig)
    y_test = y_test/max(y_orig)
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds
    censoring_train = np.logical_not(censoring_train).astype(int)
    print(sum(censoring_train)/(len(y_train)))
    return x_train[()], y_train[()], censoring_train, x_test[()], y_test[()]

def get_support():
    f1 = h5py.File("data/support_train_test.h5",'r+')   
    x_train = f1['train']['x']
    y_orig = f1['train']['t']
    censoring_train = f1['train']['e']
    x_test = f1['test']['x']
    y_test = f1['test']['t']
    y_train = y_orig/max(y_orig)
    y_test = y_test/max(y_orig)
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds
    censoring_train = np.logical_not(censoring_train).astype(int)
    print(sum(censoring_train)/(len(y_train)))
    return x_train[()], y_train[()], censoring_train, x_test[()], y_test[()]

def get_IHC4():
    f1 = h5py.File("data/metabric_IHC4_clinical_train_test.h5",'r+')   
    x_train = f1['train']['x']
    y_orig = f1['train']['t']
    censoring_train = f1['train']['e']
    x_test = f1['test']['x']
    y_test = f1['test']['t']
    y_train = y_orig/max(y_orig)
    y_test = y_test/max(y_orig)
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds
    censoring_train = np.logical_not(censoring_train).astype(int)
    print(sum(censoring_train)/(len(y_train)))
    return x_train[()], y_train[()], censoring_train, x_test[()], y_test[()]


def get_sim():
    f1 = h5py.File("data/sim_treatment_dataset.h5",'r+')   
    x_train = f1['train']['x']
    y_orig = f1['train']['t']
    censoring_train = f1['train']['e']
    x_test = f1['test']['x']
    y_test = f1['test']['t']
    y_train = y_orig/max(y_orig)
    y_test = y_test/max(y_orig)
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds
    censoring_train = np.logical_not(censoring_train).astype(int)
    print(sum(censoring_train)/(len(y_train)))
    return x_train[()], y_train[()], censoring_train, x_test[()], y_test[()]

def get_whas():
    f1 = h5py.File("data/whas_train_test.h5",'r+')   
    x_train = f1['train']['x']
    y_orig = f1['train']['t']
    censoring_train = f1['train']['e']
    x_test = f1['test']['x']
    y_test = f1['test']['t']
    y_train = y_orig/max(y_orig)
    y_test = y_test/max(y_orig)
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds
    censoring_train = np.logical_not(censoring_train).astype(int)
    print(sum(censoring_train)/(len(y_train)))
    return x_train[()], y_train[()], censoring_train, x_test[()], y_test[()]



def get_sklearn():
    rs = np.random.RandomState(seed=10)
    ns = 10000 + 1000
    nf = 10
    #x, y_orig, coef = make_regression(n_samples=ns, n_features=nf, coef=True, noise=0.0, random_state=rs)
    #x = pd.DataFrame(x)
    #y = pd.Series(y_orig)
    #x, y_orig = make_regression(n_samples=ns, n_features=nf, coef=False, noise=1.0, random_state=rs)
    x, y_orig = make_friedman1(n_samples=ns, n_features=6, noise=0.0, random_state=rs)

    y_orig = y_orig + np.random.normal(loc=0, scale=0.1*abs(x[:,2]), size=x.shape[0]) ## Homo noise
    #y_orig = (y_orig - np.min(y_orig))/(np.max(y_orig)-np.min(y_orig))
    y_orig = (y_orig - np.mean(y_orig))/(np.std(y_orig))
    y = y_orig.copy()
    n_quantiles = 3 # two-thirds of the data is truncated
    quantile = 100/float(n_quantiles)
    upper = np.percentile(y, (n_quantiles - 1) * quantile)
    right = y > upper
    censoring = np.zeros((ns,))
    censoring[right] = 1
    y = np.clip(y, a_min=None, a_max=upper)
    return split_data(x, y, y_orig, censoring, test_size = 1000, verbose = True)


def get_causalbad():
    n = 1000+10000
    x_t = np.random.normal(0, 1, size=n)
    t = np.random.binomial(size=n, n=1, p=torch.sigmoid(torch.Tensor(2*x_t+0.5)).numpy())
    t[(t== 0) | ((t==1) & (x_t < 0.0))] = 0
    #y_true = (2*t-1)*x_t+(2*t+1)-2*np.sin(2*(2*t-1)*x_t)+2*(1+0.5*x_t)+np.random.normal(0,1, size=n)
    y_true = (2*t-1)*x_t+(2*t+1)-2*np.sin(2*(2*t-1)*x_t)+2*(1+0.5*x_t)+np.random.normal(0,1+(t*0.30*x_t)+((t-1)*0.10*x_t))
    y_cens = y_true.copy()
    censoring = t
    y_cens[t==1] = y_cens[t==1]*0.7

    #tmp = (censoring== 0) | ((censoring==1) & (x_t > 0.0))
    #y_true = y_true[tmp]
    #y_cens = y_cens[tmp]
    #x_t = x_t[tmp]
    #censoring = censoring[tmp]
    #n = x_t.shape[0]
    x = x_t.reshape(n,1)
    return split_data(x, y_cens, y_true, censoring, test_size = 1000, verbose = True)