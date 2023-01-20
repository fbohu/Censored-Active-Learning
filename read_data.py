import pandas as pd
import numpy as np
import copy
import h5py
import torch
import torchvision
import os
import random
import scipy
from collections import defaultdict
from sklearn import preprocessing
from sklearn.datasets import make_regression, make_friedman1

def get_dataset(name):
    if name == 'synth':
        return get_synth()
    if name == 'churn':
        return get_churn()
    if name == 'credit_risk':
        return get_credit()
    elif name == 'gsbg':
        return get_gsbg()
    elif name == 'support':
        return get_support()
    elif name == 'IHC4':
        return get_IHC4()
    elif name == 'whas':
        return get_whas()
    elif name=='breastMSK':
        return get_bmsk()
    elif name=='mnist':
        return get_mnist()
        
def split_data(x, y_cens, y_true, censoring, test_size = 100, verbose = False):
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)
    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
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
    n = 10000
    # Define underlying function
    x = np.random.normal(5, 1.0, size=n)
    y_true = 0.5*np.sin(2*x) + 2
    # Generate noisy observations 
    y_obs = y_true + np.random.normal(loc=0, scale=0.01*abs(x), size=x.shape[0]) ## Heterogenue noise
    y_cens = copy.deepcopy(y_obs)

    # Select random points as censored and apply p% censoring
    censoring = np.int32(0.5*np.sin(2*x) + 2 >= 2.0) 
    p_c = np.random.uniform(low=0.10, high=0.30, size=np.sum(censoring==1))
    y_cens[censoring == 1] = y_obs[censoring == 1]*(1-p_c)
    #cens_levl = 2.0
    #y_cens[censoring == 1] = cens_levl + np.random.normal(loc=0, scale=0.05, size=sum(censoring))
    x = x.reshape(n,1)
    x_train, y_train, censoring_train, x_test, y_test = split_data(x, y_cens, y_true, censoring, test_size = 1000, verbose = True)
    np.random.seed(10)


    n = len(x_test)
    val_ids = np.random.choice(np.arange(0,n), size=250, replace=False)
    x_val = x_test[np.isin(np.arange(n), val_ids)]
    y_val = y_test[np.isin(np.arange(n), val_ids)]
    x_test = x_test[~np.isin(np.arange(n), val_ids)]
    y_test = y_test[~np.isin(np.arange(n), val_ids)]

    x_test = np.linspace(1.5, 8.5, 500)
    y_test = 0.5*np.sin(2*x_test) + 2
    x_test = x_test.reshape(500,1)

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_val = (x_val-means)/stds
    x_test = (x_test-means)/stds    
    y_train = (y_train-mean_y)/std_y
    y_test = (y_test-mean_y)/std_y
    y_val = (y_val-mean_y)/std_y

    print("Censoring: {}".format(sum(censoring_train)/(len(y_train))))
    print("Train: {}".format(x_train.shape))
    print("y-Train: {}".format(y_train.shape))
    print("Val: {}".format(x_val.shape))
    print("y-Val: {}".format(y_val.shape))
    print("Test: {}".format(x_test.shape))
    print("y-test: {}".format(y_test.shape))


    return x_train, y_train, censoring_train, x_val, y_val, x_test, y_test  
    
def parse_csv(dataset, features, time_column, event_column, flip =True):
    # Borrowed from square.github.io/pysurvival/ tutorials
    N_null = sum(dataset[features].isnull().sum())
    print("The raw_dataset contains {} null values".format(N_null)) #0 null values

    # Removing duplicates if there exist
    N_dupli = sum(dataset.duplicated(keep='first'))
    dataset = dataset.drop_duplicates(keep='first').reset_index(drop=True)
    print("The raw_dataset contains {} duplicates".format(N_dupli))

    # Number of samples in the dataset
    N = dataset.shape[0]
    # Building training and testing sets
    from sklearn.model_selection import train_test_split
    index_train, index_test = train_test_split(range(N), test_size = 0.35, random_state=42)
    data_train = dataset.loc[index_train].reset_index( drop = True )
    data_test  = dataset.loc[index_test].reset_index( drop = True )

    # Creating the X, T and E inputs
    x_train, x_test = data_train[features].values, data_test[features].values
    y_train, y_test = data_train[time_column].values, data_test[time_column].values
    censoring_train, _ = data_train[event_column].values, data_test[event_column].values
    if flip:
        censoring_train = np.logical_not(censoring_train).astype(int)
    
    x_train = x_train[y_train != 0]
    censoring_train = censoring_train[y_train != 0]
    y_train = y_train[y_train != 0]

    x_test = x_test[y_test != 0]
    y_test = y_test[y_test != 0]

    y_train = np.log(y_train) # log_transform
    y_test = np.log(y_test) # log_transform
    n = len(x_test)
    np.random.seed(42)
    val_ids = np.random.choice(np.arange(0,n), size=int(n*0.2), replace=False)
    x_val = x_test[np.isin(np.arange(n), val_ids)]
    y_val = y_test[np.isin(np.arange(n), val_ids)]
    x_test = x_test[~np.isin(np.arange(n), val_ids)]
    y_test = y_test[~np.isin(np.arange(n), val_ids)]

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_val = (x_val-means)/stds
    x_test = (x_test-means)/stds    

    y_test = y_test/max(y_train)
    y_val = y_val/max(y_train)
    y_train = y_train/max(y_train)

    print("Censoring: {}".format(sum(censoring_train)/(len(y_train))))
    print("Train: {}".format(x_train.shape))
    print("y-Train: {}".format(y_train.shape))
    print("Val: {}".format(x_val.shape))
    print("y-Val: {}".format(y_val.shape))
    print("Test: {}".format(x_test.shape))
    print("y-test: {}".format(y_test.shape))
    return x_train, y_train, censoring_train, x_val, y_val, x_test, y_test  


def get_churn():
    dataset = pd.read_csv("data/churn.csv")
    # Creating the time and event columns
    time_column = 'months_active'
    event_column = 'churned'
    # Extracting the features
    features = np.setdiff1d(dataset.columns, [time_column, event_column]).tolist()
    return parse_csv(dataset, features, time_column, event_column, flip=True)

def get_credit():
    dataset = pd.read_csv("data/credit_risk.csv")
    # Creating the time and event columns
    time_column = 'duration'
    event_column = 'full_repaid'
    # Extracting the features
    features = np.setdiff1d(dataset.columns, [time_column, event_column] ).tolist()
    return parse_csv(dataset, features, time_column, event_column, flip=True)


def process_h5(datasets, verbose = True, clip_outliers = False, x_lim=5.0):
    # Borrowed from github.com/TeaPearce/Censored_Quantile_Regression_NN/blob/main/01_code/datasets.py
    x_train = datasets['train']['x']
    y_orig = datasets['train']['t']
    censoring_train = datasets['train']['e']
    x_test = datasets['test']['x']
    y_test = datasets['test']['t']
    censoring_train = np.logical_not(censoring_train).astype(int)
    n = len(y_test)
    np.random.seed(10)
    val_ids = np.random.choice(np.arange(0,n), size=int(n*0.2), replace=False)
    x_val = x_test[()][np.isin(np.arange(n), val_ids)]
    y_val = y_test[()][np.isin(np.arange(n), val_ids)]
    x_test = x_test[~np.isin(np.arange(n), val_ids)]
    y_test = y_test[()][~np.isin(np.arange(n), val_ids)]

    y_test = y_test/max(y_orig)
    y_val = y_val/max(y_orig)
    y_train = y_orig/max(y_orig)    
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    x_train = (x_train-means)/stds
    x_val = (x_val-means)/stds
    x_test = (x_test-means)/stds

    if clip_outliers:
        # clip outliers
        x_train = np.clip(x_train,-x_lim,x_lim)
        x_val = np.clip(x_val,-x_lim,x_lim)
        x_test = np.clip(x_test,-x_lim,x_lim)

    if verbose:
        print("Censoring: {}".format(sum(censoring_train)/(len(y_train))))
        print("Train: {}".format(x_train.shape))
        print("y-Train: {}".format(y_train.shape))
        print("Val: {}".format(x_val.shape))
        print("y-Val: {}".format(y_val.shape))
        print("Test: {}".format(x_test.shape))
        print("y-test: {}".format(y_test.shape))

    return x_train, y_train, censoring_train, x_val, y_val, x_test, y_test

def get_gsbg():
    datasets = defaultdict(dict)
    with h5py.File("data/gbsg_cancer_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    datasets['train']['t'] = np.log(datasets['train']['t'])
    datasets['test']['t'] = np.log(datasets['test']['t'])
    return process_h5(datasets, clip_outliers=False)
    

def get_support():
    datasets = defaultdict(dict)
    with h5py.File("data/support_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    datasets['train']['t'] = np.log(datasets['train']['t'])
    datasets['test']['t'] = np.log(datasets['test']['t'])
    return process_h5(datasets, clip_outliers=False)

def get_IHC4():
    datasets = defaultdict(dict)
    with h5py.File("data/metabric_IHC4_clinical_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    datasets['train']['t'] = np.log(datasets['train']['t'])
    datasets['test']['t'] = np.log(datasets['test']['t'])
    return process_h5(datasets, clip_outliers=False)

def get_whas():
    datasets = defaultdict(dict)
    with h5py.File("data/whas_train_test.h5", 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    datasets['train']['t'] = np.log(datasets['train']['t'])
    datasets['test']['t'] = np.log(datasets['test']['t'])
    return process_h5(datasets, clip_outliers=False)


def split_tsv(x,y, test_size,censoring, clip_outliers=True, x_lim=5.0):
    # Borrowed from github.com/TeaPearce/Censored_Quantile_Regression_NN/blob/main/01_code/datasets.py
    np.random.seed(10)

    y = np.log(y+1e-7)
    test_ids = np.random.choice(np.arange(0,x.shape[0]), size=test_size, replace=False)

    x_train = x[~np.isin(np.arange(x.shape[0]), test_ids)]
    y_train = y[~np.isin(np.arange(x.shape[0]), test_ids)]
    censoring_train = censoring[~np.isin(np.arange(x.shape[0]), test_ids)]
    x_test = x[np.isin(np.arange(x.shape[0]), test_ids)]
    y_test = y[np.isin(np.arange(x.shape[0]), test_ids)]

    n = len(y_train)
    np.random.seed(10)
    val_ids = np.random.choice(np.arange(0,n), size=int(n*0.125), replace=False)
    x_val = x_train[np.isin(np.arange(n), val_ids)]
    y_val = y_train[np.isin(np.arange(n), val_ids)]
    x_train = x_train[~np.isin(np.arange(n), val_ids)]
    y_train = y_train[~np.isin(np.arange(n), val_ids)]
    censoring_train = censoring_train[~np.isin(np.arange(n), val_ids)]
    
    #normaliation
    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    x_train = (x_train-means)/stds
    x_test = (x_test-means)/stds
    x_val = (x_val-means)/stds

    y_test = y_test/max(y_train)
    y_val = y_val/max(y_train)
    y_train = y_train/max(y_train)

    if clip_outliers:
        # clip outliers
        x_train = np.clip(x_train,-x_lim,x_lim)
        x_val = np.clip(x_val,-x_lim,x_lim)
        x_test = np.clip(x_test,-x_lim,x_lim)
    print("Censoring: {}".format(sum(censoring_train)/(len(y_train))))
    print("Train: {}".format(x_train.shape))
    print("Val: {}".format(x_val.shape))
    print("Test: {}".format(x_test.shape))


    return x_train, y_train, censoring_train, x_val, y_val, x_test, y_test

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

    # there is a large value - clip it.
    x = np.array(data)
    x_lim = 50.0
    x = np.clip(x,-x_lim,x_lim)
    
    y = target[:,0]
    # in dataset 1=observed, 0=censored, so invert this
    censoring = np.abs(target[:,1]-1)
    test_size = int(data.shape[0]-(data.shape[0]*0.8)) # number of obs used for testing.
    return split_tsv(x, y, test_size, censoring)
    
def get_mnist():
    x_train, y_train, censoring_train = mnist(type_='training')
    x_test, y_test, _ = mnist(type_='test')
    y_train = np.log(y_train)
    y_test = np.log(y_test)
    n = len(y_test)
    np.random.seed(10)
    val_ids = np.random.choice(np.arange(0,n), size=5000, replace=False)
    
    x_val = x_test[np.isin(np.arange(n), val_ids)]
    y_val = y_test[np.isin(np.arange(n), val_ids)]
    x_test = x_test[~np.isin(np.arange(n), val_ids)]
    y_test = y_test[~np.isin(np.arange(n), val_ids)]

    print("Censoring: {}".format(sum(censoring_train)/(len(y_train))))
    print("Train: {}".format(x_train.shape))
    print("Val: {}".format(x_val.shape))
    print("Test: {}".format(x_test.shape))
    return x_train, y_train, censoring_train, x_val, y_val, x_test, y_test

def mnist(type_='training'):
    # datasets and opening code borrowed from github.com/TeaPearce/Censored_Quantile_Regression_NN/blob/main/01_code/datasets.py
    input_dim=(1,28,28)
    if type_  == 'training':
        np.random.seed(10)
    else: 
        np.random.seed(25)

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
    df.data, df.class_ = torch.load(os.path.join(os.path.join(os.path.join(path_data,'MNIST'),'processed'),type_ + '.pt'))
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

    df.target = np.random.gamma(shape=np.square(risks_mean)/risks_var,scale=1/(risks_mean/risks_var)) 
    # self.df.target = np.random.gamma(shape=np.square(risks_mean)/var,scale=1/(risks_mean/var)) 
    # 1/ as diff param -- see https://en.wikipedia.org/wiki/Gamma_distribution

    # normalisation
    df.data = df.data/255
    censoring_ = np.random.uniform(df.data[:,0,0,0]*0.+df.target.min(), df.data[:,0,0,0]*0.+np.quantile(df.target,0.9))
    x = np.array(df.data)
    y = df.target
    if type_  == 'training':
        censoring = (censoring_< y)*1.0 # 1 if censored else 0
        y = np.minimum(y, censoring_)
    else: 
        censoring = (censoring_ < df.target)*1.0 

    return x, y, censoring
