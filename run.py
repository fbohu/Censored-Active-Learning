import os
os.environ['TT_CUDNN_DETERMINISTIC'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from models import DenseMCDropoutNetwork
from query_strategies import random_sampling, unc, bald, censbald, duo_bald
from models.losses import tobit_nll
from read_data import *
#from datasets import get_dataset
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from tqdm.auto import trange
import pickle
import random
import scipy
tfd = tfp.distributions
import gc

def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def visual(active_ids_, start, index, name, censoring_train):
    means = start.net.sample(x_test).detach()
    loss = start.evaluate(x_test, y_test)
    scores, _ = start.get_scores(5)
    scores_exp = np.exp(scores)
    scores_exp[np.isnan(scores_exp)] = 1e-7 # used to get good results
    scores_exp = scores_exp/scores_exp.sum()
    softplus = torch.nn.Softplus()
    stds =  1e-5 + softplus(means[:,1])
    stds_ =  1e-5 + softplus(means[:,-1])

    plt.style.use('default')
    width = 487.8225
    fig, ax = plt.subplots(figsize=set_size(width*1.1,1), frameon=False)
    plt.ylim(-3.5, 7.5)
    plt.rcParams.update({'font.size': 18})
    plt.plot(x_test, means.mean(1)[:,0],color='#DB4430', label = r'$\mu_i(x_i)$')
    plt.fill_between(x_test.squeeze().numpy(), y1 = means.mean(1)[:,0]-2*stds_.mean(1),
                            y2 = means.mean(1)[:,0]+2*stds_.mean(1),alpha=0.15, color='#DB4430', label = r'$\mu_i(x_i)\pm 2 \sigma_i(x_i)$')
    plt.ylim(-3.5, 9.5)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.scatter(x_train[active_ids_][censoring_train[active_ids_] == 0], y_train[active_ids_][censoring_train[active_ids_] == 0],s=75, color='#4285F9', edgecolors='black', marker = 'o')
    plt.scatter(x_train[active_ids_][censoring_train[active_ids_] == 1], y_train[active_ids_][censoring_train[active_ids_] == 1],s=75, color='#0F9D50', edgecolors='black', marker = 'X')
    plt.plot(x_test, y_test, color='black', linestyle='dashed')
    plt.xlabel("x", fontsize=18)
    plt.ylabel("y", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("figures/synth/fit/fit_" + name +"_"+ str(index)+".pdf")
    plt.close()
    c = censoring_train[active_ids_]
    x_t = x_train[active_ids_].squeeze().numpy()
    plt.style.use('default')
    width = 487.8225
    fig, ax = plt.subplots(figsize=set_size(width*1.1,1), frameon=False)
    plt.ylim(-3.5, 7.5)
    plt.rcParams.update({'font.size': 18})
    plt.hist(x_t[c == 0], color='#4285F9', bins = np.arange(-5,5,0.5),  hatch="\\",label = 'Acquired observations')
    plt.hist(x_t[c == 1], color='#0F9D50', bins = np.arange(-5,5,0.5), alpha=0.8,  hatch="/", label = 'Acquired censored observations')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x_plot = np.linspace(xmin, xmax, 100)
    mu = 0
    std = 1.0
    p = scipy.stats.norm.pdf(x_plot, mu, std)
    plt.plot(x_plot, rescale(p, min(p), max(p), min(p), 60), color='black', label = r'$\mathcal{D}^{pool}$ distribution.')
    plt.legend(loc='upper left')
    plt.ylim((None,105))
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel("x", fontsize=18)
    plt.ylabel("y", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("figures/synth/hist/hist_" + name +"_"+ str(index)+".pdf")
    plt.close()
    

np.random.seed(42) # set seet for common active ids.
torch.manual_seed(42)
random.seed(42)

dataset = "synth"
x_train, y_train, censoring_train,x_val, y_val, x_test, y_test = get_dataset(dataset)
model_args = {'in_features': 1,
                    'out_features': 4,
                    'hidden_size':[128,128, 128],
                    'dropout_p': 0.1,
                    'epochs': 1000,
                    'lr_rate':3e-3,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'dataset':'synth',
                    'size':'synth'}

model_args_2 = {'in_features': 1,
                    'out_features': 2,
                    'hidden_size':[128,128, 128],
                    'dropout_p': 0.1,
                    'epochs': 1000,
                    'lr_rate':3e-3,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'dataset':'synth',
                    'size':'synth'}

## Params synth
init_size = 3
query_size = 3
n_rounds = 100 # The first iteration is silent is silent.
trials = 3

print(n_rounds)
print(x_train.shape)
print(y_train.shape)
print(censoring_train.shape)
print(x_test.shape)
print(y_test.shape)
print(init_size)
random = np.zeros([trials, n_rounds])
bald_ = np.zeros([trials, n_rounds])
cbald_ = np.zeros([trials, n_rounds])
unc_ = np.zeros([trials, n_rounds])
duo_ = np.zeros([trials, n_rounds])

c_random = np.zeros([trials, n_rounds])
c_bald_ = np.zeros([trials, n_rounds])
c_cbald_ = np.zeros([trials, n_rounds])
c_unc_ = np.zeros([trials, n_rounds])
c_duo_ = np.zeros([trials, n_rounds])

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
x_test = torch.from_numpy(x_test).float()
y_val = torch.from_numpy(y_val).float()
x_val = torch.from_numpy(x_val).float()
plt_threshold = 5


for k in range(0, trials):
    active_ids = np.zeros(x_train.shape[0], dtype = bool)
    ids_tmp = np.arange(x_train.shape[0])
    active_ids[np.random.choice(ids_tmp,init_size, replace=False)] = True
    active_ids_1 = active_ids.copy()
    active_ids_2 = active_ids.copy()
    active_ids_3 = active_ids.copy()
    active_ids_4 = active_ids.copy()
    active_ids_5 = active_ids.copy()

    start = duo_bald.DuoBaldSampling(x_train, y_train, censoring_train, active_ids_1, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    duo_[k,0] = start.evaluate(x_test, y_test)
    c_duo_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='duo_bald'):
        q_ids = start.query(query_size)
        active_ids_10[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_10, start, i, "duo_bald_"+str(k), censoring_train)
        start.update(active_ids_10)
        start.train()
        duo_[k,i] = start.evaluate(x_test, y_test)
        c_duo_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = unc.UncertaintySampling(x_train, y_train, censoring_train, active_ids_2, model_args_2, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    unc_[k,0] = start.evaluate(x_test, y_test)
    unc_c[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='avg'):
        q_ids = start.query(query_size)
        active_ids_6[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_6, start, i, "unc_"+str(k), censoring_train)
        start.update(active_ids_6)
        start.train()
        unc_[k,i] = start.evaluate(x_test, y_test)
        unc_c[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = censbald.CensBaldSampling(x_train, y_train, censoring_train, active_ids_3, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    cbald_[k,0] = start.evaluate(x_test, y_test)
    c_cbald_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='cbald'):
        q_ids = start.query(query_size)
        active_ids_9[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_9, start, i, "cbald_"+str(k), censoring_train)
        start.update(active_ids_9)
        start.train()
        cbald_[k,i] = start.evaluate(x_test, y_test)
        c_cbald_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = bald.BaldSampling(x_train, y_train, censoring_train, active_ids_4, model_args_2, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    bald_[k,0] = start.evaluate(x_test, y_test)
    c_bald_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='bald'):
        q_ids = start.query(query_size)
        active_ids_1[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_1, start, i, "bald_"+str(k), censoring_train)
        start.update(active_ids_1)
        start.train()
        bald_[k,i] = start.evaluate(x_test, y_test)
        c_bald_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = random_sampling.RandomSampling(x_train, y_train, censoring_train, active_ids_5, model_args_2, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    random[k,0] =start.evaluate(x_test, y_test)
    c_random[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='random'):
        q_ids = start.query(query_size)
        active_ids_2[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_2, start, i, "random_"+str(k), censoring_train)
        start.update(active_ids_2)
        start.train()
        random[k,i] = start.evaluate(x_test, y_test)
        c_random[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
        

a = {'random': random,
    'bald':bald_,
    'cbald':cbald_,
    #'mu':mu_,
    #'mupi':muclass,
    'pi':class_,
    'rho':avg,
    #'tau':tau_,
    'murho':muavg,
    'mutatu': duo_}

with open('results/' + dataset + '_filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

a = {'random': c_random,
    'bald':c_bald_,
    'cbald':c_cbald_,
    'mu':c_mu_,
    'pi':c_class,
    'rho':c_avg,
    'murho':c_muavg,
    'mutatu': c_duo_}

with open('results/' + dataset + '_censored_filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
