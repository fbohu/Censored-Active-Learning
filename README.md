# Bayesian Active Learning for Censored Modelling and Survival Analysis

This work reproduces the experiments for our paper: Bayesian Active Learning for Censored Modelling and Survival Analysis.

# Experiments
To run the experiment `python3 main.py --dataset NAME_OF_DATA_SET --model "normal" --query QUERY_NAME --init_size 5 --query_size 3 --num_trials 25 --n_rounds 100  --hidden_size 128 --layers 3`


# Queries
Here is a list of the names for the different queries (Paper name: code name)
- Random : 'random'
- Uncertainty : 'unc'
- BALD : 'bald'
- C-BALD : 'cbald'
- T-BALD : 'duobald'


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
