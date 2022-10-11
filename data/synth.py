import numpy as np
from numpy.random import RandomState
from scipy.stats import norm


def generate_synthetic_data(verbose, betas, seed, epsilon, size):
    prng = RandomState(seed=seed)
    x0 = np.ones(size)
    x1 = np.power(-1, prng.binomial(n=1, p=0.5, size=size))
    x2 = prng.normal(size=size)
    X = np.concatenate([x0, x1, x2]).reshape(3, size)
    noise = epsilon(X=X, prng=prng, size=size)
    y_star = np.dot(np.array(betas), X) + noise
    y = np.clip(y_star, a_min=0, a_max=None)
    if verbose:
        print('Censored observations: %d of %d (%.2f%%)' % (sum(y == 0), size, sum(y == 0) / size))
    return {'x0': x0, 'x1': x1, 'x2': x2, 'noise': noise, 'X': X, 'y': y, 'y_star': y_star}


def make_ds1(verbose, num_samples, seed):
    return generate_synthetic_data(
        verbose=verbose,
        size=num_samples,
        betas=(1, 1, 1), 
        seed=seed, 
        epsilon=epsilon_ds1)


def epsilon_ds1(X, prng, size):
    return prng.normal(size=size)
        

def ppf_ystar_ds1(ds, theta):
    return norm(loc=ds['x0'] + ds['x1'] + ds['x2'],
                scale=1)\
        .ppf(theta)


def ppf_censored_y_ds1(ds, theta):
    return np.maximum(0, ppf_ystar_ds1(ds, theta))

def make_ds2(verbose, num_samples, seed):
    return generate_synthetic_data(
        verbose=verbose,
        size=num_samples, 
        betas=(1, 1, 1), 
        seed=seed, 
        epsilon=epsilon_ds2)


def epsilon_ds2(X, prng, size):
    return np.multiply((1 + X[2, :]), prng.normal(size=size))


def ppf_ystar_ds2(ds, theta):
    return norm(loc=ds['x0'] + ds['x1'] + ds['x2'], 
                scale=abs(1 + ds['x2']))\
        .ppf(theta)


def ppf_censored_y_ds2(ds, theta):
    return np.maximum(0, ppf_ystar_ds2(ds, theta))


def make_ds3(verbose, num_samples, seed):
    return generate_synthetic_data(
        verbose=verbose,
        size=num_samples, 
        betas=(1, 1, 1), 
        seed=seed, 
        epsilon=epsilon_ds3)


def epsilon_ds3(X, prng, size):
    return (0.75 * prng.normal(size=size)) + (0.25 * prng.normal(size=size, scale=2))


def ppf_ystar_ds3(ds, theta):
    return norm(loc=ds['x0'] + ds['x1'] + ds['x2'], 
                scale=((0.75 ** 2) + (0.25 ** 2)) ** 0.5)\
        .ppf(theta)


def ppf_censored_y_ds3(ds, theta):
    return np.maximum(0, ppf_ystar_ds3(ds, theta))