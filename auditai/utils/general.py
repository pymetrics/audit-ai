import numpy as np
from scipy.stats import norm
from scipy.special import gammaln


def two_tailed_ztest(success1, success2, total1, total2):
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
        z score for two tailed z-test
    p_value : float
        p value for two tailed z-test
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


def dirichln(arr):
    """
    Dirichlet gamma function
    Albert (2007) Bayesian Computation with R, 1st ed., pg 178

    Parameters
    ----------
    arr : array or matrix of float values

    Returns
    -------
    val : float or array,
        logged Dirichlet transformed value if array or matrix
    """
    val = np.sum(gammaln(arr)) - gammaln(np.sum(arr))
    return val


def get_unique_name(new_name, name_list, addendum='_new'):
    """
    Utility function to return a new unique name if name is in list.

    Parameters
    ----------
    new_name : string
        name to be updated
    name_list: list
        list of existing names
    addendum: string
        addendum appended to new_name if new_name is in name_list

    Returns
    -------
    new_name : string,
        updated name

    Example
    -------
    new_name = 'feat1'
    name_list = ['feat1', 'feat2']

    first iteration: new_name returned = 'feat1_new'
    now with name_list being updated to include new feature:
        name_list = ['feat1', 'feat2', 'feat1_new']

    second iteration: new_name returned = 'feat1_new_new'
    """
    # keep appending "new" until new_name is not in list
    while new_name in name_list:
        new_name += addendum
    return new_name
