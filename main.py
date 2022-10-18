import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from models import DenseMCDropoutNetwork
from query_strategies import random_sampling, bald, mu, mupi, pi, rho, tau, murho
from models.losses import tobit_nll
from read_data import *

def get_strat(which):
    return {
        'random': random_sampling.RandomSampling,
        'bald': bald.BaldSampling,
    }[which]


def main(args):
    strat = get_strat(args.query)
    x_train, y_train, censoring_train, x_test, y_test = get_dataset(args.dataset)
    model_args = {'in_features': x_train.shape[-1],
            'hidden_size':[16]}

    model_performance = np.zeros([trials, n_rounds])
    censored = np.zeros([trials, n_rounds])


    np.random.seed(1) # set seet for common active ids.
    for k in range(0, args.num_trials):
        active_ids = np.zeros(x_train.shape[0], dtype = bool)
        ids_tmp = np.arange(x_train.shape[0])
        active_ids[np.random.choice(ids_tmp,init_size, replace=False)] = True


        start = strat(x_train, y_train, censoring_train, active_ids, model_args)
        start.train()
        model_performance[k,0] = start.evaluate(x_test, y_test)
        censored[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
        for i in range(1,args.n_rounds):
            q_ids = start.query(query_size)
            active_ids[q_ids] = True
            start.update(active_ids)
            start.train()
            model_performance[k,i] = start.evaluate(x_test, y_test)
            censored[k,i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
        
    
    results = {'model_perf': model_performance,
                'censored': censored}
    with open('results/'+ args.query + "-" + dataset + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    os.environ['TT_CUDNN_DETERMINISTIC'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='ds1')
    parser.add_argument('--query', type=str, default='random')
    parser.add_argument('--num_trials', type=int, default = 1)
    parser.add_argument('--n_rounds', type=int, default = 10)

    args = parser.parse_args()
    print(args)
    main(args)