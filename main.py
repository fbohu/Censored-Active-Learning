import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import argparse
import random
from tqdm.auto import trange

from query_strategies import random_sampling, bald, censbald, duo_bald, avg_bald, class_bald
from read_data import *

def get_strat(which):
    return {
        'random': random_sampling.RandomSampling,
        'bald': bald.BaldSampling,
        'cbald': censbald.CensBaldSampling,
        'duobald': duo_bald.DuoBaldSampling,
        'avg_bald': avg_bald.AvgBaldSampling,
        'classbald': class_bald.ClassBaldSampling,
    }[which]

def get_model(which, x_train):
    return {'small': {'in_features': x_train.shape[-1],
                    'out_features': 4,
                    #'hidden_size':[128,128],
                    'hidden_size':[32,32],
                    'dropout_p': 0.25,
                    'epochs': 1000,
                    'lr_rate':3e-4,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            'medium' :{'in_features': x_train.shape[-1],
                    'out_features': 4,
                    'hidden_size':[128,128,128],
                    'dropout_p': 0.25,
                    'epochs': 1000,
                    'lr_rate':3e-4,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            'big' :{'in_features': x_train.shape[-1],
                    'out_features': 4,
                    'hidden_size':[128,128,128,128],
                    'dropout_p': 0.25,
                    'epochs': 1000,
                    'lr_rate':3e-4,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            }[which]

def main(args):
    results_path = "results/" + args.dataset + "/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    strat = get_strat(args.query)
    x_train, y_train, censoring_train, x_test, y_test = get_dataset(args.dataset)
    model_args = get_model(args.model, x_train)

    model_performance = np.zeros([args.num_trials, args.n_rounds+1])
    censored = np.zeros([args.num_trials, args.n_rounds+1])

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    x_test = torch.from_numpy(x_test).float()
     
    torch.use_deterministic_algorithms(True)
    #for k in trange(0, args.num_trials, desc='number of trials'):
    for k in range(0, args.num_trials):

        np.random.seed(123+k) # set seet for common active ids.
        torch.manual_seed(123+k)
        random.seed(123+k)   
        active_ids = np.zeros(x_train.shape[0], dtype = bool)
        ids_tmp = np.arange(x_train.shape[0])
        active_ids[np.random.choice(ids_tmp, args.init_size, replace=False)] = True
        start = strat(x_train, y_train, censoring_train, active_ids, model_args, random_seed=k)
        start.train()
        model_performance[k, 0] = start.evaluate(x_test, y_test)
        censored[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
        for i in range(1,args.n_rounds+1):
        #for i in trange(1, args.n_rounds+1, desc='running rounds'):        
            q_ids = start.query(args.query_size)
            active_ids[q_ids] = True
            start.update(active_ids)
            start.train()
            model_performance[k, i] = start.evaluate(x_test, y_test)
            censored[k, i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
        
        print(f"Done with {k} out of {args.num_trials}")
    
    results = {'model_perf': model_performance,
                'censored': censored}
    with open(results_path + args.query + "-" + args.model + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='synth')
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--query', type=str, default='bald')
    parser.add_argument('--init_size', type=int, default=2)
    parser.add_argument('--query_size', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default = 2)
    parser.add_argument('--n_rounds', type=int, default = 2)

    args = parser.parse_args()
    print(args)
    main(args)