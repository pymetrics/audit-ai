import numpy as np
from pandas import DataFrame
import scipy as sc

from .utils.crosstabs import (crosstab,
                              crosstab_bayes_factor,
                              crosstab_df,
                              crosstab_odds_ratio,
                              crosstab_ztest,
                              get_top_bottom_indices,
                              top_bottom_crosstab)
from .utils.validate import (check_consistent_length,
                             check_array,
                             boolean_array)


def ztest(labels, results, threshold=None):
    """
    Two-tailed z score for proportions

    Parameters
    ----------
    labels : array_like
        categorical labels for each corresponding value of `result` ie. M/F

    results : array_like
        binary decision values, if continuous values are supplied then
        the `threshold` must also be supplied to generate binary decisions

    threshold : numeric
        value dividing scores into True/False

    Returns
    -------
    z-score: float
        test statistic of two-tailed z-test
        z > 1.960 is signficant at p = .05
        z > 2.575 is significant at p = .01
        z > 3.100 is significant at p = .001
    """

    check_consistent_length(labels, results)
    results = check_array(results)

    # convert the results to True/False
    results = boolean_array(results, threshold=threshold)
    # get crosstab for two groups
    ctab = top_bottom_crosstab(labels, results)

    zstat, pvalue = crosstab_ztest(ctab)
    return zstat, pvalue


def fisher_exact(labels, results, threshold=None):
    """
    Returns odds ratio and p-value of Fisher's exact test
    Uses scipy.stats.fisher_exact, which only works for 2x2 contingency tables
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html

    Parameters
    ----------
    labels : array_like
        categorical labels for each corresponding value of `result` ie. M/F

    results : array_like
        binary decision values, if continuous values are supplied then
        the `threshold` must also be supplied to generate binary decisions

    threshold : numeric
        value dividing scores into True/False, where result>=threshold == True

    Returns
    -------
    oddsratio : float
        This is prior odds ratio and not a posterior estimate.
    pvalue : float
        P-value, the probability of obtaining a distribution at least
        as extreme as the one that was actually observed, assuming that
        the null hypothesis is true.
    """

    check_consistent_length(labels, results)
    results = check_array(results)

    # convert the results to True/False
    results = boolean_array(results, threshold=threshold)
    # get crosstab for two groups
    ctab = top_bottom_crosstab(labels, results)

    oddsratio, pvalue = sc.stats.fisher_exact(ctab)
    return oddsratio, pvalue


def chi2(labels, results, threshold=None):
    """
    Returns odds ratio and p-value of Chi-square test of independence
    Uses scipy.stats.chi2_contingency, using an Rx2 contingency table
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

    Parameters
    ----------
    labels : array_like
        categorical labels for each corresponding value of `result` ie. M/F

    results : array_like
        binary decision values, if continuous values are supplied then
        the `threshold` must also be supplied to generate binary decisions

    threshold : numeric
        value dividing scores into True/False, where result>=threshold == True

    Returns
    -------
    chi2_stat : float
        The test statistic.
    pvalue : float
        P-value, the probability of obtaining a distribution at least
        as extreme as the one that was actually observed, assuming that
        the null hypothesis is true.
    """

    check_consistent_length(labels, results)
    results = check_array(results)

    # convert the results to True/False
    results = boolean_array(results, threshold=threshold)
    ctab = crosstab(labels, results)

    chi2_stat, pvalue = sc.stats.chi2_contingency(ctab)[:2]
    return chi2_stat, pvalue


def bayes_factor(labels, results, threshold=None,
                 priors=None, top_bottom=True):
    """
    Computes the analytical Bayes factor for an nx2 contingency table.
    If matrix is bigger than 2x2, calculates the bayes factor for the entire
    dataset unless top_bottom is set to True. Will always calculate odds ratio
    between largest and smallest diverging groups


    Adapted from Albert (2007) Bayesian Computation with R,
        1st ed., pg 178:184

    Parameters
    ----------
    labels : array_like
        categorical labels for each corresponding value of `result` ie. M/F

    results : array_like
        binary decision values, if continuous values are supplied then
        the `threshold` must also be supplied to generate binary decisions

    threshold : numeric, optional
        value dividing scores into True/False, where result>=threshold == True

    priors : ndarray, optional, shape (n,2)
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
    odds_ratio : float,
        odds ratio for the highest and lowest groups.

    Examples
    --------
    TODO : update examples with odds ratios

    >>> from biastesting.categorical import bf_ctabs
    >>> ctabs = np.array([[50,50],[50,50]]) # 1:1 hit/miss ratio
    >>> # assuming uniformity
    >>> uniform_prior = np.array([[1,1],[1,1]])
    >>> bf_ctabs(ctabs, uniform_prior) #all same leads to low BF (dependence)
        0.26162148804907587

    >>> biased_prior =  np.array([[5,10],[70,10]])
    >>> # all same given strong priors lowers BF (stronger proof of dependence)
    >>> bf_ctabs(ctabs, biased_prior)
        0.016048077654948357

    >>> biased_ctab = np.array([[75,50],[25,50]])
    >>> # biased crosstabs against prior assumption of uniformity
    >>> # large BF (strong proof of independence)
    >>> bf_ctabs(biased_ctab, biased_prior)
        202.95548692414306

    >>> # assuming a large prior bias in one direction
    >>> # conteracts a large observed dataset in the opposite direction
    >>> bf_ctabs(biased_ctab, biased_prior)
        0.00012159518854984268
    """
    check_consistent_length(labels, results)
    results = check_array(results)

    # convert the results to True/False
    results = boolean_array(results, threshold=threshold)
    ctab = crosstab_df(labels, results)

    if top_bottom:
        used_indices = list(get_top_bottom_indices(ctab))
        ctab = ctab.iloc[used_indices]

    else:
        used_indices = list(ctab.index)

    if priors:
        prior = DataFrame(priors)
        prior = prior.iloc[used_indices]
        prior = prior.values
    else:
        prior = np.ones(shape=ctab.shape)

    ctab = ctab.values

    BF = crosstab_bayes_factor(ctab, prior)
    oddsratio = crosstab_odds_ratio(ctab)

    return oddsratio, BF
