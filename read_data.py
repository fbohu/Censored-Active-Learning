import pandas as pd
import numpy as np
import copy
import h5py
import torch
import torchvision
import os
from collections import defaultdict
from data.synth import *
from sklearn import preprocessing
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
    elif name=='tmb':
        return get_tmb()
    elif name=='breastMSK':
        return get_bmsk()
    elif name=='lgggbm':
        return get_lgggbm()
    elif name=='mnist':
        return get_mnist()
    elif name == 'cbald':
        return get_causalbad()
        


def split_data(x, y_cens, y_true, censoring, test_size = 100, verbose = False):
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)
    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    print(x.shape[0])
    print(test_ids.shape)
    y_train = y_cens[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y_true[np.isin(np.arange(x.shape[0]), test_ids)]
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds


    y_train = (y_train-mean_y)/std_y
    y_test = (y_test-mean_y)/std_y

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
    x = np.linspace(0, 10, n)
    y_true = 0.5*np.sin(2*x) + 2 #+ x/10
    #y_true = 0.5*x + 2

    # Generate noisy observations 
    y_obs = y_true + np.random.normal(loc=0, scale=0.01*x, size=x.shape[0]) ## Heterogenue noise
    #y_obs = y_true + np.random.normal(loc=0, scale=0.01, size=x.shape[0]) ## Homo noise
    y_cens = copy.deepcopy(y_obs)

    # Select random points as censored and apply p% censoring
    censoring = np.int32(0.5*np.sin(2*x) + 2 >= 2.2) 
    #censoring = np.random.choice(2, n, p=[0.1, 0.9])*censoring # this can be used to uncensor some.
    p_c = np.random.uniform(low=0.20, high=0.30, size=np.sum(censoring==1))
    y_cens[censoring == 1] = y_obs[censoring == 1]*(1-p_c)
    #y_cens[censoring == 1] = cens_levl + np.random.normal(loc=0, scale=0.01, size=sum(censoring))

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
    datasets = defaultdict(dict)
    with h5py.File("data/gbsg_cancer_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    x_train = datasets['train']['x']
    y_orig = datasets['train']['t']
    censoring_train = datasets['train']['e']
    x_test = datasets['test']['x']
    y_test = datasets['test']['t']
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
    datasets = defaultdict(dict)
    with h5py.File("data/support_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    x_train = datasets['train']['x']
    y_orig = datasets['train']['t']
    censoring_train = datasets['train']['e']
    x_test = datasets['test']['x']
    y_test = datasets['test']['t']

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
    datasets = defaultdict(dict)
    with h5py.File("data/metabric_IHC4_clinical_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    x_train = datasets['train']['x']
    y_orig = datasets['train']['t']
    censoring_train = datasets['train']['e']
    x_test = datasets['test']['x']
    y_test = datasets['test']['t']

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
    datasets = defaultdict(dict)
    with h5py.File("data/sim_treatment_dataset.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    x_train = datasets['train']['x']
    y_orig = datasets['train']['t']
    censoring_train = datasets['train']['e']
    x_test = datasets['test']['x']
    y_test = datasets['test']['t']

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
    datasets = defaultdict(dict)
    with h5py.File("data/whas_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    x_train = datasets['train']['x']
    y_orig = datasets['train']['t']
    censoring_train = datasets['train']['e']
    x_test = datasets['test']['x']
    y_test = datasets['test']['t']

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
    n = 2500+10000
    np.random.seed(10)
    x_t = np.random.normal(0, 1, size=n)
    t = np.random.binomial(size=n, n=1, p=torch.sigmoid(torch.Tensor(2*x_t+0.5)).numpy())
    #t[(t== 0) | ((t==1) & (x_t < -0.25))] = 0
    y_true = (2*t-1)*x_t+(2*t+1)-2*np.sin(2*(2*t-1)*x_t)+2*(1+0.5*x_t)+np.random.normal(0,1, size=n)
    #y_true = (2*t-1)*x_t+(2*t+1)-2*np.sin(2*(2*t-1)*x_t)+2*(1+0.5*x_t)+np.random.normal(0,1+(t*0.30*x_t)+((1-t)*0.10*abs(x_t)))
    y_cens = y_true.copy()
    censoring = t
    y_cens[t==1] = y_cens[t==1]*0.7

    x = x_t.reshape(n,1)
    return split_data(x, y_cens, y_true, censoring, test_size = 2500, verbose = True)


def get_tmb():
    df=pd.read_table('data/tmb_mskcc_2018_clinical_data.tsv',sep='\t')
    event_arr = np.array(df['Overall Survival Status'])
    df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
    df['time'] = np.array(df['Overall Survival (Months)']).astype(np.float)

    df['age_new'] = np.array(df['Age at Which Sequencing was Reported (Days)'])
    sex_arr = np.array(df['Sex'])
    df['sex_new'] = np.array([1 if sex_arr[i] == 'Female' else 0 for i in range(sex_arr.shape[0])])

    # remove nans
    df.dropna(subset = ['event', 'time', 'age_new','sex_new','TMB (nonsynonymous)'],how='any',inplace=True)

    # use self.target instead of self.df.target to avoid pandas warning
    target = pd.concat([df.pop(x) for x in ['time','event']], axis=1)
    data = pd.concat([df.pop(x) for x in ['age_new','sex_new','TMB (nonsynonymous)']], axis=1)

    target = np.array(target)
    x = np.array(data)

    # clip outliers - Considering clipping data to remove outliers.
    #x_lim = 5
    #x = np.clip(x,-x_lim,x_lim)
    #target = np.clip(target,-x_lim,x_lim)

    censoring= target[:,1]
    y = target[:,0]

    # in dataset 1=observed, 0=censored, so invert this
    target[:,1] = np.abs(target[:,1]-1)
    test_size = int(data.shape[0]-(data.shape[0]*0.8)) # number of obs used for testing.

    np.random.seed(10)
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)

    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    y_train = y[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y[np.isin(np.arange(x.shape[0]), test_ids)]
    
    #normaliation
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds

    y_train = (y_train-mean_y)/std_y
    y_test = (y_test-mean_y)/std_y


    
    return x_train, y_train, censoring_train, x_test, y_test


def get_bmsk():
    df=pd.read_table('data/breast_msk_2018_clinical_data.tsv',sep='\t')

    event_arr = np.array(df['Overall Survival Status'])
    df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
    df['time'] = df['Overall Survival (Months)']

    tmp_arr = np.array(df['ER Status of the Primary'])
    df['ER_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])
    tmp_arr = np.array(df['Overall Patient HER2 Status'])
    df['HER2_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])
    tmp_arr = np.array(df['Overall Patient HR Status'])
    df['HR_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])

    # remove nans
    df.dropna(subset = ['event', 'time', 'ER_new','HER2_new','HR_new','Mutation Count','TMB (nonsynonymous)'],how='any',inplace=True)

    # use self.target instead of self.df.target to avoid pandas warning
    target = pd.concat([df.pop(x) for x in ['time','event']], axis=1)
    data = pd.concat([df.pop(x) for x in ['ER_new','HER2_new','HR_new','Mutation Count','TMB (nonsynonymous)']], axis=1)

    target = np.array(target)
    data = np.array(data)

    input_dim=data.shape[1]

    # in dataset 1=observed, 0=censored, so invert this
    target[:,1] = np.abs(target[:,1]-1)

    # there is a large value - clip it.
    x = np.array(data)
    x_lim = 50
    x = np.clip(x,-x_lim,x_lim)
    censoring= target[:,1]
    y = target[:,0]

    # in dataset 1=observed, 0=censored, so invert this
    target[:,1] = np.abs(target[:,1]-1)
    test_size = int(data.shape[0]-(data.shape[0]*0.8)) # number of obs used for testing.
    np.random.seed(10)
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)

    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    y_train = y[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y[np.isin(np.arange(x.shape[0]), test_ids)]
    
    #normaliation
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds

    y_train = (y_train-mean_y)/std_y
    y_test = (y_test-mean_y)/std_y
    
    return x_train, y_train, censoring_train, x_test, y_test



def get_lgggbm():
    df=pd.read_table('data/lgggbm_tcga_pub_clinical_data.tsv',sep='\t')

    # remove nans
    df.dropna(subset = ['Overall Survival Status', 'Overall Survival (Months)', 'Diagnosis Age','Sex','Absolute Purity','Mutation Count','TMB (nonsynonymous)'],how='any',inplace=True)

    event_arr = np.array(df['Overall Survival Status'])
    df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
    df['time'] = np.array(df['Overall Survival (Months)']).astype(np.float)

    tmp_arr = np.array(df['Sex'])
    df['sex_new'] = np.array([1 if tmp_arr[i] == 'Female' else 0 for i in range(tmp_arr.shape[0])])

    # use self.target instead of self.df.target to avoid pandas warning
    target = pd.concat([df.pop(x) for x in ['time','event']], axis=1)
    data = pd.concat([df.pop(x) for x in ['Diagnosis Age','sex_new','Absolute Purity','Mutation Count','TMB (nonsynonymous)']], axis=1)

    target = np.array(target)
    data = np.array(data)

    # in dataset 1=observed, 0=censored, so invert this
    target[:,1] = np.abs(target[:,1]-1)

    x = np.array(data)
    x_lim = 125
    x = np.clip(x,-x_lim,x_lim)

    censoring= target[:,1]
    y = target[:,0]

    # in dataset 1=observed, 0=censored, so invert this
    target[:,1] = np.abs(target[:,1]-1)
    test_size = int(data.shape[0]-(data.shape[0]*0.8)) # number of obs used for testing.
    np.random.seed(10)
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)

    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    y_train = y[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y[np.isin(np.arange(x.shape[0]), test_ids)]
    
    #normaliation
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds

    y_train = (y_train-mean_y)/std_y
    y_test = (y_test-mean_y)/std_y
    
    return x_train, y_train, censoring_train, x_test, y_test


def get_mnist():
    input_dim=(1,28,28)

    # download this. must use version torchvision==0.9.1 to get processed folder
    # https://github.com/pytorch/vision/issues/4685
    path_data = "data/"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,), (0.5,)),
                                                ])
    trainset = torchvision.datasets.MNIST(path_data, download=True, train=True, transform=transform)
    # trainset is of shape (60000 , 2), with x and y (int)
    # each x example is (1, 28, 28)

    # we'll match structure of other RealTargetSyntheticCensor datasets by basing around a df object
    class Object(object):
        pass
    df = Object()

    # load whole dataset
    df.data, df.class_ = torch.load(os.path.join(os.path.join(os.path.join(path_data,'MNIST'),'processed'),'training.pt'))
    df.data, df.class_ = df.data.numpy(), df.class_.numpy() # this is also dumb but needed for censoring gen
    df.data = np.expand_dims(df.data,1) # add an extra channel

    # we now generate targets and censoring as per 
    # App D.2 https://arxiv.org/pdf/2101.05346.pdf
    # basically, each class is assigned a risk group, which has an associated risk score
    # this parameterises a gamma dist from which targets are drawn
    risk_list = [11.25, 2.25, 5.25, 5.0, 4.75, 8.0, 2.0, 11.0, 1.75, 10.75]
    var_list = [0.1, 0.5, 0.1, 0.2, 0.2, 0.2, 0.3, 0.1, 0.4, 0.6]
    # var_list = [1e-3]*10
    risks_mean = np.zeros_like(df.class_)+0.9
    risks_var = np.zeros_like(df.class_)+0.9
    for i in range(10):
        risks_mean[df.class_==i] = risk_list[i]
        risks_var[df.class_==i] = var_list[i]

    np.random.seed(10)
    df.target = np.random.gamma(shape=np.square(risks_mean)/risks_var,scale=1/(risks_mean/risks_var)) 
    # self.df.target = np.random.gamma(shape=np.square(risks_mean)/var,scale=1/(risks_mean/var)) 
    # 1/ as diff param -- see https://en.wikipedia.org/wiki/Gamma_distribution

    # normalisation
    df.data = df.data/255
    df.target = df.target/10
    censoring_ = np.random.uniform(df.data[:,0,0,0]*0.+df.target.min(), df.data[:,0,0,0]*0.+np.quantile(df.target,0.9))
    censoring = (censoring_ < df.target)*1.0 
    x = np.array(df.data)
    y = df.target
    test_size = int(df.data.shape[0]-(df.data.shape[0]*0.8)) # number of obs used for testing.
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)

    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    y_train = y[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y[np.isin(np.arange(x.shape[0]), test_ids)]
    
    return x_train, y_train, censoring_train, x_test, y_test    
