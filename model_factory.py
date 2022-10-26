model_args = {
    'small': {'in_features': x_train.shape[-1],
            'out_features': 4,
            'hidden_size':[128,128],
            'dropout_p': 0.25,
            'epochs': 500,
            'lr_rate':3e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    'med' :{'in_features': x_train.shape[-1],
            'out_features': 4,
            'hidden_size':[128,128,128],
            'dropout_p': 0.25,
            'epochs': 500,
            'lr_rate':3e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    'big' :{'in_features': x_train.shape[-1],
            'out_features': 4,
            'hidden_size':[128,128,128,128],
            'dropout_p': 0.25,
            'epochs': 500,
            'lr_rate':3e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
}