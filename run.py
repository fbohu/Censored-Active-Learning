import os
os.environ['TT_CUDNN_DETERMINISTIC'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from models import DenseMCDropoutNetwork
from query_strategies import random_sampling, bald, mu, mupi, pi, rho, tau, murho, mutau, censbald, duo_bald, avg_bald, class_bald
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
    fig, ax = plt.subplots(figsize=set_size(width,1), frameon=False)
    #fig = plt.figure(figsize=(12,6))
    #ax = fig.add_subplot(111)
    plt.ylim(-3.5, 7.5)
    plt.rcParams.update({'font.size': 16})
    #plt.title(str(loss))
    #for i in range(0,20):
    #    plt.plot(x_test, means[:,i,0],'bo', alpha=0.01, zorder=0)
        #plt.plot(x_test, means[:,i,0]+2*stds_[:,i],'bo', alpha=0.01, zorder=0)
        #plt.plot(x_test, means[:,i,0]-2*stds_[:,i],'bo', alpha=0.01, zorder=0)

    #    plt.plot(x_test, means[:,i,2],'ro', alpha=0.01, zorder=0)
        #plt.plot(x_test, means[:,i,2]+2*stds[:,i],'ro', alpha=0.01, zorder=0)
        #plt.plot(x_test, means[:,i,2]-2*stds[:,i],'ro', alpha=0.01, zorder=0)
        #plt.plot(x, -1*samples[i,:,-1],'ro', alpha=0.01)
    plt.plot(x_test, means.mean(1)[:,0],color='#DB4430', label = 'Probabilistic fit')
    #plt.plot(x_test, means.mean(1)[:,0]+2*stds_.mean(1),label='Mean from ensemble', color='blue')
    #plt.plot(x_test, means.mean(1)[:,0]-2*stds_.mean(1),label='Mean from ensemble',  color='blue'
    plt.fill_between(x_test.squeeze().numpy(), y1 = means.mean(1)[:,0]-2*stds_.mean(1),
                            y2 = means.mean(1)[:,0]+2*stds_.mean(1),alpha=0.15, color='#DB4430')
    #plt.plot(x_test, means.mean(1)[:,2], label='Mean from ensemble',  color='red', zorder=2)
    #plt.plot(x_test, means.mean(1)[:,2]-2*stds.mean(1),label='Mean from ensemble',  color='red', zorder=2)
    #plt.plot(x_test, means.mean(1)[:,2]+2*stds.mean(1),label='Mean from ensemble', color='red', zorder=2)
    #plt.xlim(1,9)
    plt.ylim(-3.5, 7.5)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.scatter(x_train[active_ids_][censoring_train[active_ids_] == 0], y_train[active_ids_][censoring_train[active_ids_] == 0],s=75, color='#4285F9', edgecolors='black', marker = 'o')
    plt.scatter(x_train[active_ids_][censoring_train[active_ids_] == 1], y_train[active_ids_][censoring_train[active_ids_] == 1],s=75, color='#0F9D50', edgecolors='black', marker = 'X')
    plt.plot(x_test, y_test, color='black', linestyle='dashed')
    #plt.scatter(x_train[active_ids_][censoring_train[active_ids_] == 0], y_train[active_ids_][censoring_train[active_ids_] == 0], color='black', zorder=3, s=50)
    #plt.scatter(x_train[active_ids_][censoring_train[active_ids_] == 1], y_train[active_ids_][censoring_train[active_ids_] == 1], marker="x", color='red', zorder=3, s=50)
    #plt.scatter(x_train.numpy()[q_ids], y_train.numpy()[q_ids],color='green', zorder=3, s=100)
    #plt.ylim(-3.5, 6.5)
    #plt.xlim(-2,2)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("figures/cbald/fit/fit_" + name +"_"+ str(index)+".pdf")
    plt.close()
    c = censoring_train[active_ids_]
    x_t = x_train[active_ids_].squeeze().numpy()
    #print(c)
    #print(x_t.shape)
    plt.style.use('default')
    width = 487.8225
    fig, ax = plt.subplots(figsize=set_size(width,1), frameon=False)
    #fig = plt.figure(figsize=(12,6))
    #ax = fig.add_subplot(111)
    plt.ylim(-3.5, 7.5)
    plt.rcParams.update({'font.size': 16})
    plt.hist(x_t[c == 0], color='#4285F9', bins = np.arange(-5,5,0.5),  hatch="\\",label = 'Acquired observations')
    plt.hist(x_t[c == 1], color='#0F9D50', bins = np.arange(-5,5,0.5), alpha=0.8,  hatch="/", label = 'Acquired censored observations')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x_plot = np.linspace(xmin, xmax, 100)
    mu = 0
    std = 1.0
    p = norm.pdf(x_plot, mu, std)
    plt.plot(x_plot, rescale(p, min(p), max(p), min(p), 60), color='black', label = 'x distribution')
    plt.legend(loc='upper left')
    plt.ylim((None,95))
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/cbald/hist/hist_" + name +"_"+ str(index)+".pdf")
    plt.close()
    selected = list(active_ids_.copy())
    #selected.extend(q_ids.copy())
    plt.figure(figsize=(16,8))
    #plt.plot(x_train[~np.array(selected)], scores + scipy.stats.gumbel_r.rvs(
    #        loc=0, scale=0.25, size=len(scores), random_state=None),'bo',zorder=0, alpha=0.1)
    #plt.plot(x_train[~np.array(selected)], scores, 'ro',zorder=1)
    plt.plot(x_train[~np.array(selected)], scores_exp, 'bo',zorder=1)
    plt.ylim(0,None)
    plt.savefig("figures/cbald/scores/scores_" + name +"_"+ str(index)+".png")
    plt.close()
    

np.random.seed(1) # set seet for common active ids.
torch.manual_seed(1)
random.seed(1)

dataset = "synth"
x_train, y_train, censoring_train,x_val, y_val, x_test, y_test = get_dataset(dataset)
model_args = {'in_features': 1,
                    'out_features': 4,
                    #'hidden_size':[256,256, 256,256],
                    'hidden_size':[128,128, 128],
                    'dropout_p': 0.1,
                    'epochs': 1000,
                    'lr_rate':3e-3,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'dataset':'synth',
                    'size':'synth'}
## Params ds1, ds2, ds3, 
#init_size = 10
#query_size = 3
#n_rounds = 30 # The first iteration is silent is silent.
#trials = 1

## Params cbald
#init_size = 100
#query_size = 10
#n_rounds = 10 # The first iteration is silent is silent.
#trials = 1


## Params synth
init_size = 5
query_size = 3
n_rounds = 100 # The first iteration is silent is silent.
trials = 3


## Params sklearn
#init_size = 25
#query_size = 5
#n_rounds = 100 # The first iteration is silent is silent.
#trials = 5

#init_size = 25
#query_size = 3
#n_rounds = 125#int((x_train.shape[0]-init_size)/query_size)#100 # The first iteration is silent is silent.
#trials = 5
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
mu_ = np.zeros([trials, n_rounds])
muclass = np.zeros([trials, n_rounds])
class_ = np.zeros([trials, n_rounds])
avg = np.zeros([trials, n_rounds])
tau_ = np.zeros([trials, n_rounds])
muavg = np.zeros([trials, n_rounds])
duo_ = np.zeros([trials, n_rounds])

c_random = np.zeros([trials, n_rounds])
c_bald_ = np.zeros([trials, n_rounds])
c_cbald_ = np.zeros([trials, n_rounds])
c_mu_ = np.zeros([trials, n_rounds])
c_muclass = np.zeros([trials, n_rounds])
c_class = np.zeros([trials, n_rounds])
c_avg = np.zeros([trials, n_rounds])
c_tau_ = np.zeros([trials, n_rounds])
c_muavg = np.zeros([trials, n_rounds])
c_duo_ = np.zeros([trials, n_rounds])

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
x_test = torch.from_numpy(x_test).float()
y_val = torch.from_numpy(y_val).float()
x_val = torch.from_numpy(x_val).float()
plt_threshold = 4


for k in trange(0, trials, desc='number of trials'):
    active_ids = np.zeros(x_train.shape[0], dtype = bool)
    #active_ids[np.random.choice(np.where((x < 3.0) | (x > 5.6))[0], init_size, replace=False)] = True
    ids_tmp = np.arange(x_train.shape[0])
    active_ids[np.random.choice(ids_tmp,init_size, replace=False)] = True
    active_ids_1 = active_ids.copy()
    active_ids_2 = active_ids.copy()
    active_ids_3 = active_ids.copy()
    active_ids_4 = active_ids.copy()
    active_ids_5 = active_ids.copy()
    active_ids_6 = active_ids.copy()
    active_ids_7 = active_ids.copy()
    active_ids_8 = active_ids.copy()
    active_ids_9 = active_ids.copy()
    active_ids_10 = active_ids.copy()

    '''
    start = murho.MuRhoSampling(x_train, y_train, censoring_train, active_ids_8, model_args, random_seed=k)
    start.train()
    murho_[k,0] = start.evaluate(x_test, y_test)
    c_murho_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='murho'):
        q_ids = start.query(query_size)
        active_ids_8[q_ids] = True
        if (k == 0) and (dataset == 'synth'):
            visual(active_ids_8, start, i, "murho_"+str(k), censoring_train)
        start.update(active_ids_8)
        start.train()
        murho_[k,i] = start.evaluate(x_test, y_test)
        c_murho_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
    start = tau.TauSampling(x_train, y_train, censoring_train, active_ids_7, model_args, random_seed=k)
    start.train()
    tau_[k,0] = start.evaluate(x_test, y_test)
    c_tau_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='tau'):
        q_ids = start.query(query_size)
        active_ids_7[q_ids] = True
        if (k == 0) and (dataset == 'synth'):
            visual(active_ids_7, start, i, "tau_"+str(k), censoring_train)
        start.update(active_ids_7)
        start.train()
        tau_[k,i] = start.evaluate(x_test, y_test)
        c_tau_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
    '''
    start = duo_bald.DuoBaldSampling(x_train, y_train, censoring_train, active_ids_10, model_args, x_val=x_val, y_val=y_val, random_seed=k)
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
    '''
    start = avg_bald.AvgBaldSampling(x_train, y_train, censoring_train, active_ids_6, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    avg[k,0] = start.evaluate(x_test, y_test)
    c_avg[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='avg'):
        q_ids = start.query(query_size)
        active_ids_6[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_6, start, i, "avg_"+str(k), censoring_train)
        start.update(active_ids_6)
        start.train()
        avg[k,i] = start.evaluate(x_test, y_test)
        c_avg[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = class_bald.ClassBaldSampling(x_train, y_train, censoring_train, active_ids_5, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    class_[k,0] = start.evaluate(x_test, y_test)
    c_class[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='class_bald'):
        q_ids = start.query(query_size)
        active_ids_5[q_ids] = True
        if (k < plt_threshold) and (dataset == 'synth'):
            visual(active_ids_5, start, i, "class_bald_"+str(k), censoring_train)
        start.update(active_ids_5)
        start.train()
        class_[k,i] = start.evaluate(x_test, y_test)
        c_class[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
    start = mupi.MuPiSampling(x_train, y_train, censoring_train, active_ids_4, model_args, random_seed=k)
    start.train()
    muclass[k,0] = start.evaluate(x_test, y_test)
    c_muclass[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds,desc='mupi'):
        q_ids = start.query(query_size)
        active_ids_4[q_ids] = True
        if (k == 0) and (dataset == 'synth'):
            visual(active_ids_4, start, i, "mupi"+str(k), censoring_train)
        start.update(active_ids_4)
        start.train()
        muclass[k,i] = start.evaluate(x_test, y_test)
        c_muclass[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
    
    start = mu.MuSampling(x_train, y_train, censoring_train, active_ids_3, model_args, random_seed=k)
    start.train()
    mu_[k,0] = start.evaluate(x_test, y_test)
    c_mu_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds,desc='mu'):
        q_ids = start.query(query_size)
        active_ids_3[q_ids] = True
        if (k == 0) and (dataset == 'synth'):
            visual(active_ids_3, start, i, "mu"+str(k), censoring_train)
        start.update(active_ids_3)
        start.train()
        mu_[k,i] = start.evaluate(x_test, y_test)
        c_mu_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
    '''
    start = censbald.CensBaldSampling(x_train, y_train, censoring_train, active_ids_9, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    cbald_[k,0] = start.evaluate(x_test, y_test)
    c_cbald_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
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

    start = bald.BaldSampling(x_train, y_train, censoring_train, active_ids_1, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    bald_[k,0] = start.evaluate(x_test, y_test)
    c_bald_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
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

    start = random_sampling.RandomSampling(x_train, y_train, censoring_train, active_ids_2, model_args, x_val=x_val, y_val=y_val, random_seed=k)
    start.train()
    random[k,0] =start.evaluate(x_test, y_test)
    c_random[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
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
    #'mupi':c_muclass,
    'pi':c_class,
    'rho':c_avg,
    #'tau':c_tau_,
    'murho':c_muavg,
    'mutatu': c_duo_}

with open('results/' + dataset + '_censored_filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


plt.close()
plt.figure(figsize=(16,8))
plt.plot(np.mean(random, axis=0),'-.', label='Random', linewidth=3)
plt.plot(np.mean(bald_,axis=0),'-.', label='Bald')
plt.plot(np.mean(cbald_,axis=0),'-.', label='cBald')
#plt.plot(np.mean(mu_,axis=0),'-.', label='mu_')
#plt.plot(np.mean(muclass,axis=0),'-.', label='muclass')
plt.plot(np.mean(class_,axis=0),'-.', label='class')
plt.plot(np.mean(avg,axis=0),'-.', label='avg')
#plt.plot(np.mean(tau_,axis=0),'-.', label='tau_')
#plt.plot(np.mean(muavg,axis=0),'-.', label='muavg')
plt.plot(np.mean(duo_,axis=0),'-.', label='duo_')
plt.legend()
#plt.ylim(None, 0.5)
plt.savefig("figures/"+dataset + "results.png")
plt.close()

plt.close()
plt.figure(figsize=(16,8))
plt.plot(np.mean(c_random,axis=0),'-.', label='Random', linewidth=3)
plt.plot(np.mean(c_bald_,axis=0),'-.', label='Bald')
plt.plot(np.mean(c_cbald_,axis=0),'-.', label='cBald')
#plt.plot(np.mean(c_mu_,axis=0),'-.', label='mu_')
#plt.plot(np.mean(c_muclass,axis=0),'-.', label='muclass')
plt.plot(np.mean(c_class,axis=0),'-.', label='class')
plt.plot(np.mean(c_avg,axis=0),'-.', label='avg')
#plt.plot(np.mean(c_tau_,axis=0),'-.', label='tau_')
#plt.plot(np.mean(c_muavg,axis=0),'-.', label='muavg')
plt.plot(np.mean(c_duo_,axis=0),'-.', label='duo_')
plt.legend(loc='lower right')
plt.savefig("figures/"+dataset+ "results_censoring.png")
plt.close()
