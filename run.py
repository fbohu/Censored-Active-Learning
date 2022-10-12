import os
os.environ['TT_CUDNN_DETERMINISTIC'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models import DenseMCDropoutNetwork
from query_strategies import random_sampling, bald, mu, mupi, pi, rho, tau, murho, mutau, censbald
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
tfd = tfp.distributions
import gc

def visual(active_ids, start, index, name):
    mean = start.net.sample(x_test).detach()
    plt.figure(figsize=(16,8))
    plt.plot(y_test, color='black')
    for i in range(0,20):
        plt.plot(mean[i,:,0],'bo', alpha=0.01)
        plt.plot(mean[i,:,2],'ro', alpha=0.01)
        #plt.plot(x, -1*samples[i,:,-1],'ro', alpha=0.01)
    plt.plot(mean.mean(1)[:,0], label='Mean from ensemble', color='blue')
    plt.plot(mean.mean(1)[:,2], label='Mean from ensemble', color='orange')
    #plt.scatter(x_test, y_test,color='red')
    #plt.scatter(x, y_obs, color='black')
    #plt.scatter(x, y_cens, color='red')
    #plt.scatter(x_train[active_ids], y_train[active_ids], color='green')
    #plt.scatter(x_train.numpy()[q_ids], y_train.numpy()[q_ids],color='blue')
    plt.savefig("figures/sklearn/" + name + str(index)+".png")
    plt.close()
    plt.figure(figsize=(16,8))
    plt.plot(y_test,mean.mean(1)[:, 0],'o')
    plt.plot(np.arange(-5,30,1),np.arange(-5,30,1))
    plt.ylim(-4,4)
    plt.xlim(-4,4)
    plt.savefig("figures/sklearn/scatter_" + name + str(index)+".png")
    plt.close()


dataset = "sklearn"
x_train, y_train, censoring_train, x_test, y_test = get_dataset(dataset)
model_args = {'in_features': x_train.shape[-1],
            'hidden_size':[100,100]}


## Params ds1
#init_size = 5
#query_size = 3
#n_rounds = 25 # The first iteration is silent is silent.
#trials = 10


## Params sklearn
init_size = 50
query_size = 5
n_rounds = 50 # The first iteration is silent is silent.
trials = 1

#init_size = 146
#query_size = 5
#n_rounds = 30 # The first iteration is silent is silent.
#trials = 1
print(x_train.shape)
print(y_train.shape)
print(censoring_train.shape)
print(x_test.shape)
print(y_test.shape)
print(init_size)
np.random.seed(123) # set seet for common active ids.
torch.manual_seed(0)
random = np.zeros([trials, n_rounds])
bald_ = np.zeros([trials, n_rounds])
cbald_ = np.zeros([trials, n_rounds])
mu_ = np.zeros([trials, n_rounds])
mupi_ = np.zeros([trials, n_rounds])
pi_ = np.zeros([trials, n_rounds])
rho_ = np.zeros([trials, n_rounds])
tau_ = np.zeros([trials, n_rounds])
murho_ = np.zeros([trials, n_rounds])
mutau_ = np.zeros([trials, n_rounds])

c_random = np.zeros([trials, n_rounds])
c_bald_ = np.zeros([trials, n_rounds])
c_cbald_ = np.zeros([trials, n_rounds])
c_mu_ = np.zeros([trials, n_rounds])
c_mupi_ = np.zeros([trials, n_rounds])
c_pi_ = np.zeros([trials, n_rounds])
c_rho_ = np.zeros([trials, n_rounds])
c_tau_ = np.zeros([trials, n_rounds])
c_murho_ = np.zeros([trials, n_rounds])
c_mutau_ = np.zeros([trials, n_rounds])

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
x_test = torch.from_numpy(x_test).float()


for k in trange(0, trials, desc='number of trials'):
    active_ids = np.zeros(x_train.shape[0], dtype = bool)
    #active_ids[np.random.choice(np.where((x < 3.0) | (x > 5.6))[0], init_size, replace=False)] = True
    ids_tmp = np.arange(x_train.shape[0])
    active_ids[np.random.choice(ids_tmp,init_size, replace=False)] = True
    active_ids_2 = active_ids.copy()
    active_ids_3 = active_ids.copy()
    active_ids_4 = active_ids.copy()
    active_ids_5 = active_ids.copy()
    active_ids_6 = active_ids.copy()
    active_ids_7 = active_ids.copy()
    active_ids_8 = active_ids.copy()
    active_ids_9 = active_ids.copy()
    active_ids_10 = active_ids.copy()

    start = mutau.MuTauSampling(x_train, y_train, censoring_train, active_ids_10, model_args)
    start.train()
    mutau_[k,0] = start.evaluate(x_test, y_test)
    c_mutau_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='murho'):
        q_ids = start.query(query_size)
        active_ids_10[q_ids] = True
        #visual(active_ids_8, start, i, "murho_"+str(k))
        start.update(active_ids_10)
        start.train()
        mutau_[k,i] = start.evaluate(x_test, y_test)
        c_mutau_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = murho.MuRhoSampling(x_train, y_train, censoring_train, active_ids_8, model_args)
    start.train()
    murho_[k,0] = start.evaluate(x_test, y_test)
    c_murho_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    for i in trange(1,n_rounds, desc='murho'):
        q_ids = start.query(query_size)
        active_ids_8[q_ids] = True
        #visual(active_ids_8, start, i, "murho_"+str(k))
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
        #visual(active_ids_7, start, i, "tau_"+str(k))
        start.update(active_ids_7)
        start.train()
        tau_[k,i] = start.evaluate(x_test, y_test)
        c_tau_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = rho.RhoSampling(x_train, y_train, censoring_train, active_ids_6, model_args)
    start.train()
    rho_[k,0] = start.evaluate(x_test, y_test)
    c_rho_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='rho'):
        q_ids = start.query(query_size)
        active_ids_6[q_ids] = True
        #visual(active_ids_6, start, i, "rho_"+str(k))
        start.update(active_ids_6)
        start.train()
        rho_[k,i] = start.evaluate(x_test, y_test)
        c_rho_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = pi.PiSampling(x_train, y_train, censoring_train, active_ids_5, model_args)
    start.train()
    pi_[k,0] = start.evaluate(x_test, y_test)
    c_pi_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='pi'):
        q_ids = start.query(query_size)
        active_ids_5[q_ids] = True
        #visual(active_ids_5, start, i, "pi_"+str(k))
        start.update(active_ids_5)
        start.train()
        pi_[k,i] = start.evaluate(x_test, y_test)
        c_pi_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
        
    start = mupi.MuPiSampling(x_train, y_train, censoring_train, active_ids_4, model_args)
    start.train()
    mupi_[k,0] = start.evaluate(x_test, y_test)
    c_mupi_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds,desc='mupi'):
        q_ids = start.query(query_size)
        active_ids_4[q_ids] = True
        #visual(active_ids_4, start, i, "mupi"+str(k))
        start.update(active_ids_4)
        start.train()
        mupi_[k,i] = start.evaluate(x_test, y_test)
        c_mupi_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
    
    start = mu.MuSampling(x_train, y_train, censoring_train, active_ids_3, model_args)
    start.train()
    mu_[k,0] = start.evaluate(x_test, y_test)
    c_mu_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds,desc='mu'):
        q_ids = start.query(query_size)
        active_ids_3[q_ids] = True
        #visual(active_ids_3, start, i, "mu"+str(k))
        start.update(active_ids_3)
        start.train()
        mu_[k,i] = start.evaluate(x_test, y_test)
        c_mu_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = censbald.CensBaldSampling(x_train, y_train, censoring_train, active_ids_9, model_args, random_seed=k)
    start.train()
    cbald_[k,0] = start.evaluate(x_test, y_test)
    c_cbald_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='cbald'):
        q_ids = start.query(query_size)
        active_ids_9[q_ids] = True
        #visual(active_ids_9, start, i, "cbald"+str(k))
        start.update(active_ids_9)
        start.train()
        cbald_[k,i] = start.evaluate(x_test, y_test)
        c_cbald_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = bald.BaldSampling(x_train, y_train, censoring_train, active_ids, model_args, random_seed=k)
    start.train()
    bald_[k,0] = start.evaluate(x_test, y_test)
    c_bald_[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='bald'):
        q_ids = start.query(query_size)
        active_ids[q_ids] = True
        #visual(active_ids, start, i, "bald"+str(k))
        start.update(active_ids)
        start.train()
        bald_[k,i] = start.evaluate(x_test, y_test)
        c_bald_[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()

    start = random_sampling.RandomSampling(x_train, y_train, censoring_train, active_ids_2, model_args, random_seed=k)
    start.train()
    random[k,0] =start.evaluate(x_test, y_test)
    c_random[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    #print(results[k,0])
    for i in trange(1,n_rounds, desc='random'):
        q_ids = start.query(query_size)
        active_ids_2[q_ids] = True
        #visual(active_ids_2, start, i, "random"+str(k))
        start.update(active_ids_2)
        start.train()
        random[k,i] = start.evaluate(x_test, y_test)
        c_random[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
    del start
    gc.collect()
        

a = {'random': random,
    'bald':bald_,
    'cbald':cbald_,
    'mu':mu_,
    'mupi':mupi_,
    'pi':pi_,
    'rho':rho_,
    'tau':tau_,
    'murho':murho_,
    'mutatu': mutau_}

with open('results/' + dataset + 'long_filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

a = {'random': c_random,
    'bald':c_bald_,
    'cbald':c_cbald_,
    'mu':c_mu_,
    'mupi':c_mupi_,
    'pi':c_pi_,
    'rho':c_rho_,
    'tau':c_tau_,
    'murho':c_murho_,
    'mutatu': c_mutau_}

with open('results/' + dataset + '_censored_filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


plt.close()
plt.figure(figsize=(16,8))
plt.plot(np.mean(random, axis=0),'-.', label='Random', linewidth=3)
plt.plot(np.mean(bald_,axis=0),'-.', label='Bald')
plt.plot(np.mean(cbald_,axis=0),'-.', label='cBald')
plt.plot(np.mean(mu_,axis=0),'-.', label='mu_')
plt.plot(np.mean(mupi_,axis=0),'-.', label='mupi_')
plt.plot(np.mean(pi_,axis=0),'-.', label='pi_')
plt.plot(np.mean(rho_,axis=0),'-.', label='rho_')
plt.plot(np.mean(tau_,axis=0),'-.', label='tau_')
plt.plot(np.mean(murho_,axis=0),'-.', label='murho_')
plt.plot(np.mean(mutau_,axis=0),'-.', label='mutau_')
plt.legend()
#plt.ylim(None, 0.5)
plt.savefig("figures/"+dataset + "results.png")
plt.close()

plt.close()
plt.figure(figsize=(16,8))
plt.plot(np.mean(c_random,axis=0),'-.', label='Random', linewidth=3)
plt.plot(np.mean(c_bald_,axis=0),'-.', label='Bald')
plt.plot(np.mean(c_cbald_,axis=0),'-.', label='cBald')
plt.plot(np.mean(c_mu_,axis=0),'-.', label='mu_')
plt.plot(np.mean(c_mupi_,axis=0),'-.', label='mupi_')
plt.plot(np.mean(c_pi_,axis=0),'-.', label='pi_')
plt.plot(np.mean(c_rho_,axis=0),'-.', label='rho_')
plt.plot(np.mean(c_tau_,axis=0),'-.', label='tau_')
plt.plot(np.mean(c_murho_,axis=0),'-.', label='murho_')
plt.plot(np.mean(c_mutau_,axis=0),'-.', label='mutau_')
plt.legend(loc='lower right')
plt.savefig("figures/"+dataset+ "results_censoring.png")
plt.close()
