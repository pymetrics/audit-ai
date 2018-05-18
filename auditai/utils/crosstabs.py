"""
Collection of functions to create and manipulate cross tabulations
"""
import numpy as np
import pandas as pd

from .functions import _ztest, _odds_ratio, dirichln


def crosstab_df(labels, decisions):
    """
    Parameters
    ------------
    labels : array_like
        containing categorical values like ['M', 'F']
    scores : array_like
        containing real numbers

    Returns
    --------
    crosstab : 2x2 array
        in the form,
                    False True
        TopGroup       5    4
        BottomGroup    3    4
        so, crosstab = array([[5, 4], [3, 4]])
    """
    labels, decisions = pd.Series(labels), pd.Series(decisions)
    # rows are label values (e.g. ['F', 'M'])
    # columns are decision values (e.g. [False, True])
    ctab = pd.crosstab(labels, decisions)
    return ctab


def crosstab(labels, decisions):
    """
    Returns value of crosstab_df as numpy array
    """
    ctab = crosstab_df(labels, decisions)
    return ctab.values


def top_bottom_crosstab(labels, decisions):
    """
    Utility function for grabbing crosstabs for groups with highest
    and lowest pass rates, as defined by:
        - a categorical comparision column
        - a binary decision column

    Parameters
    -------
    labels : array_like
        categorical labels for each corresponding value of `decision` ie. M/F

    decision : array_like
        binary decision values, ie. True/False or 0/1

    Returns
    ---------
    top_bottom_ctabs : 2x2 array
        contains True/False frequency counts of 2 groups with highest
        and lowest pass rates
    """

    ctab = crosstab_df(labels, decisions)

    # if already 2x2, return table
    if ctab.shape == (2, 2):
        return ctab.values

    top, bottom = get_top_bottom_indices(ctab)

    top_bottom_ctab = ctab.iloc[[top, bottom]].values
    return top_bottom_ctab


def get_top_bottom_indices(ctab_df):
    """
    Utility function for returning row indices of groups
    with highest and lowest pass rates, according to input crosstab dataframe

    Parameters
    -----------
    ctab_df - pandas DataFrame, shape (n,2)
        the rows should correspond to groups, columns to binary labels

    Returns
    ---------
    top_group_idx - int
        index of group in ctab_df with highest pass rate (ratio of True/total)
    bottom_group_idx - int
        index of group in ctab_df with lowest pass rate (ratio of True/total)
    """

    if ctab_df.shape[1] != 2:
        ctab_df = ctab_df.T

    normed_ctabs = ctab_df.div(ctab_df.sum(axis=1), axis=0)

    # assume larger decision value corresponds to passing
    # e.g. max({1,0}) or max({True, False})
    true_val = ctab_df.columns.max()

    # grab top and bottom group indices
    top_group_idx = normed_ctabs[true_val].values.argmax()
    bottom_group_idx = normed_ctabs[true_val].values.argmin()
    return top_group_idx, bottom_group_idx


def crosstab_bayes_factor(ctab, prior=None):
    """
    Computes the analytical Bayes factor for an nx2 contingency table.

    Adapted from Albert (2007) Bayesian Computation with R,
        1st ed., pg 178:184

    Parameters
    ----------
    ctab : ndarray, shape (n,2)
        contingency table containing label x result frequency counts

    prior : ndarray, optional, shape (n,2)
        e.g. uniform_prior = np.array([[1,1],[1,1]])


    Returns
    -------

    BF : float,
        bayes factor for the hypothesis of independence
            BF < 1 : support for a hypothesis of dependence
            1 < BF < 3 :  marginal support for independence
            3 < BF < 20 : moderate support for independence
            20 < BF < 150 : strong support for independence
            BF > 150 : very strong support for independence


    Examples
    --------
    >>> ctab = np.array([[50,50],[50,50]]) # 1:1 hit/miss ratio
    >>> # assuming uniformity
    >>> uniform_prior = np.ones(ctab.shape)
    >>> bf_ctabs(ctab, uniform_prior) #all same leads to low BF (dependence)
        0.26162148804907587

    """
    if prior is None:
        prior = np.ones(shape=ctab.shape)

    ac = prior.sum(axis=0)  # column sums
    ar = prior.sum(axis=1)  # row sums
    yc = ctab.sum(axis=0)  # column sums
    yr = ctab.sum(axis=1)  # row sums

    I, J = ctab.shape
    OR = np.ones(I)  # vector of ones for rows
    OC = np.ones(J)  # vector of ones for columns

    lbf = dirichln(ctab + prior) + dirichln(ar - (J - 1) * OR) \
        + dirichln(ac - (I - 1) * OC) - dirichln(prior) \
        - dirichln(yr + ar - (J - 1) * OR) - dirichln(yc + ac - (I - 1) * OC)

    # calculating bayes factor for all groups
    BF = np.exp(lbf)
    return BF


def crosstab_odds_ratio(ctab):
    """
    Calculate exact odds ratio between two groups.
    Designed for 2x2 matrices. When greater than 2x2,
    calculates OR for highest and lowest groups.

    Parameters
    ----------
    ctab : matrix of values, shape(2,2)

    Returns
    -------
    odds_ratio : exact odds ratio

    Examples
    --------

    TODO: add examples
    """
    ctab = np.array(ctab, dtype=np.float)

    if ctab.shape[1] != 2:
        raise ValueError("Must be of shape (n,2)")

    if ctab.shape[0] != 2:
        tab_ratios = ctab[:, 0] / ctab[:, 1]
        max_idx = np.argmax(tab_ratios)
        min_idx = np.argmin(tab_ratios)
        ctab = ctab[:, [min_idx, max_idx]]

    a_n = ctab[0, 0]
    a_p = ctab[0, 1]
    b_n = ctab[1, 0]
    b_p = ctab[1, 1]

    oddsratio = _odds_ratio(a_n, a_p, b_n, b_p)

    return oddsratio


def crosstab_ztest(ctab):
    """
    z-scores from 2x2 cross tabs of passing rate across groups

    Parameters
    ----------
    ctab: array, shape=(2,2);
        crosstab of passing rate across groups, where each row is a group
        and the first and second columns count the number of unsuccessful
        and successful trials respectively

    Returns
    -------
    zstat: float
        test statistic for two tailed z-test
    """
    ctab = np.asarray(ctab)

    n1 = ctab[0].sum()
    n2 = ctab[1].sum()
    pos1 = ctab[0, 1]
    pos2 = ctab[1, 1]
    zstat, p_value = _ztest(pos1, pos2, n1, n2)
    return zstat, p_value
