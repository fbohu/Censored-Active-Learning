# Bayesian Active Learning for Censored Modelling and Survival Analysis

This work reproduces the experiments for our paper: Bayesian Active Learning for Censored Modelling and Survival Analysis.

# Experiments
To run the experiment `python main.py --dataset NAME_OF_DATA_SET --model "normal" --query QUERY_NAME --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`

To replicate the results from a real world data set the following scripts should be run. The results are stored in the results folder.
`python main.py --dataset IHC4 --model "normal" --query random --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`

`python main.py --dataset IHC4 --model "normal" --query unc --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`

`python main.py --dataset IHC4 --model "normal" --query bald --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`

`python main.py --dataset IHC4 --model "normal" --query cbald --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`

`python main.py --dataset IHC4 --model "normal" --query duobald --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`

## Synthetic experiment
Generation of figures for the synthetic experiment can be generated with `python3 run.py`

# Scoring functions
Here is a list of the names for the different queries (Paper name: code name)
- Random : 'random'
- Uncertainty : 'unc'
- BALD : 'bald'
- C-BALD : 'cbald'
- T-BALD : 'duobald'

The specific scoring functions can be seen in `query_stragies/`

# Model
The model is created based on the hidden_size and the layers parameters. 
Model output is defined by the query.
If a convolutional model is needed for SurvMNIST, set the dataset flag to be MNIST and the model to mnist

# Dataset
Name of datasets
- gsbg
- support
- IHC4
- whas
- breastMSK
- churn
- credit_risk
- mnist 
