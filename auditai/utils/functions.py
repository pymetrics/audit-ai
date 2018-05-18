import numpy as np
from scipy.stats import norm
from scipy.special import gammaln


def _ztest(success1, success2, total1, total2):
    """
    Two-tailed z score for proportions

    Parameters
    -------
    success1 : int
        the number of success in `total1` trials/observations

    success2 : int
        the number of success in `total2` trials/observations

    total1 : int
        the number of trials or observations of class 1

    total2 : int
        the number of trials or observations of class 2

    Returns
    -------
    zstat : float
        test statistic for two tailed z-test
    """
    p1 = success1 / float(total1)
    p2 = success2 / float(total2)
    p_pooled = (success1 + success2) / float(total1 + total2)

    obs_ratio = (1. / total1 + 1. / total2)
    var = p_pooled * (1 - p_pooled) * obs_ratio

    # calculate z-score using foregoing values
    zstat = (p1 - p2) / np.sqrt(var)

    # calculate associated p-value for 2-tailed normal distribution
    p_value = norm.sf(abs(zstat)) * 2

    return zstat, p_value


def _odds_ratio(a_n, a_p, b_n, b_p):
    """
    doc it
    """
    a_ratio = float(a_n) / a_p
    b_ratio = float(b_n) / b_p
    return a_ratio / b_ratio


def dirichln(a):
    """
    Dirichlet gamma function
    Albert (2007) Bayesian Computation with R, 1st ed., pg 178

    Parameters
    ----------
    a : array or matrix of float values

    Returns
    -------
    val : float or array,
        logged Dirichlet transformed value if array or matrix
    """
    val = np.sum(gammaln(a)) - gammaln(np.sum(a))
    return val


def get_unique_name(new_name, name_list, addendum='_new'):
    """Utility function for returning a name not contained in a list"""
    while new_name in name_list:
        new_name += addendum
    return new_name
