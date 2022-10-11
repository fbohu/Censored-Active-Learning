from scipy import stats
import numpy as np

_eps = 1e-7

def random(mu_0, mu_1, t, pt, temperature):
    return np.ones_like(mu_0.mean(1))

def tau(mu_0, mu_1, t, pt, temperature):
    return (1 / temperature) * np.log((mu_1 - mu_0).var(1) + _eps)


def mu(mu_0, mu_1, t, pt, temperature):
    return (1 / temperature) * np.log((t * mu_1.var(1) + (1 - t) * mu_0.var(1)) + _eps)


def rho(mu_0, mu_1, t, pt, temperature):
    return tau(mu_0, mu_1, t, pt, temperature) - mu(mu_0, mu_1, 1 - t, pt, temperature)


def mu_rho(mu_0, mu_1, t, pt, temperature):
    return mu(mu_0, mu_1, t, pt, temperature) + rho(mu_0, mu_1, t, pt, temperature)


def pi(mu_0, mu_1, t, pt, temperature):
    return np.log((t * (1 - pt) + (1 - t) * pt) + _eps)


def mu_pi(mu_0, mu_1, t, pt, temperature):
    return mu(mu_0, mu_1, t, pt, temperature) + pi(mu_0, mu_1, t, pt, temperature)