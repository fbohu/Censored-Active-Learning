import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import argparse
import random
from tqdm.auto import trange

from query_strategies import random_sampling, unc, bald, censbald, duo_bald
from read_data import *

def get_strat(which):
    return {
        'random': random_sampling.RandomSampling,
        'bald': bald.BaldSampling,
        'unc': unc.UncertaintySampling,
        'cbald': censbald.CensBaldSampling,
        'duobald': duo_bald.DuoBaldSampling,
    }[which]

def get_model(which, args, x_train):
    return {'normal': {'in_features': x_train.shape[-1],
                    'out_features': 2 if ((args.query == 'unc') or (args.query == 'bald') or (args.query == 'random')) else 5,
                    'hidden_size': np.repeat([args.hidden_size], args.layers),
                    'dropout_p': args.dropout,
                    'epochs': 1000,
                    'lr_rate':3e-4,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'size': str(args.hidden_size) + str(args.layers)},
            'mnist' :{'in_features': -1,
                    'out_features': 2 if ((args.query == 'unc') or (args.query == 'bald') or (args.query == 'random')) else 5,
                    'hidden_size':[128,128,128,128],
                    'dropout_p': 0.20,
                    'epochs': 1000,
                    'lr_rate':3e-4,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'size': 'mnist'},
            'synth' :{'in_features': x_train.shape[-1],
                    'out_features': 2 if ((args.query == 'unc') or (args.query == 'bald') or (args.query == 'random')) else 5,
                    'hidden_size':[128,128, 128],
                    'dropout_p': 0.20,
                    'epochs': 1000,
                    'lr_rate':3e-3,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'size': 'synth'},
            }[which]

def main(args):
    results_path = "results/" + args.dataset + "/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    strat = get_strat(args.query)
    x_train, y_train, censoring_train, x_val, y_val, x_test, y_test, censoring_test = get_dataset(args.dataset)
    model_args = get_model(args.model, args, x_train)
    model_args['dataset'] = args.dataset
    print(model_args)

    model_performance = np.zeros([args.num_trials, args.n_rounds+1])
    model_tobit_nll = np.zeros([args.num_trials, args.n_rounds+1])
    model_mae = np.zeros([args.num_trials, args.n_rounds+1])
    model_mae_hinge = np.zeros([args.num_trials, args.n_rounds+1])
    model_mae_combined = np.zeros([args.num_trials, args.n_rounds+1])
    censored = np.zeros([args.num_trials, args.n_rounds+1])

    x_train = torch.from_numpy(x_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    x_test = torch.from_numpy(x_test).float()
    for k in range(0, args.num_trials):
        np.random.seed(123+k) # set seet for common active ids.
        torch.manual_seed(123+k)
        random.seed(123+k)   

        active_ids = np.zeros(x_train.shape[0], dtype = bool)
        ids_tmp = np.arange(x_train.shape[0])
        active_ids[np.random.choice(ids_tmp, args.init_size, replace=False)] = True
        start = strat(x_train, y_train, censoring_train, active_ids, model_args, x_val=x_val, y_val=y_val, random_seed=k)
        start.train()
        model_performance[k, 0],model_tobit_nll[k,0], model_mae[k,0], model_mae_hinge[k,0], model_mae_combined[k,0] = start.evaluate(x_test, y_test, censoring_test)
        censored[k,0] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
        for i in range(1,args.n_rounds+1):      
            q_ids = start.query(args.query_size)
            active_ids[q_ids] = True
            start.update(active_ids)
            start.train()
            model_performance[k, i],model_tobit_nll[k,i], model_mae[k,i], model_mae_hinge[k,i], model_mae_combined[k,i]  = start.evaluate(x_test, y_test, censoring_test)
            censored[k, i] = np.sum(start.Cens[start.ids])/len(start.Cens[start.ids])
        
        print(f"Done with {k} out of {args.num_trials}")
    
    results = {'model_perf': model_performance,
                'model_tobit':model_tobit_nll,
                'model_mae': model_mae,
                'model_mae_hinge': model_mae_hinge,
                'model_mae_combined': model_mae_combined,
                'censored': censored}
                
    with open(results_path + args.query + "-" + model_args['size'] + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='synth')
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)

    parser.add_argument('--query', type=str, default='bald')
    parser.add_argument('--init_size', type=int, default=2)
    parser.add_argument('--query_size', type=int, default= 1)
    parser.add_argument('--num_trials', type=int, default = 2)
    parser.add_argument('--n_rounds', type=int, default = 2)

    args = parser.parse_args()
    print(args)
    main(args)