from collections import defaultdict, OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

from .utils.functions import get_unique_name
from .utils.validate import ClassifierWrapper


def classifier_posterior_probabilities(clf, df, feature_names, categories,
                                       low=None, high=None, num=100):
    """
    Function to compute posterior probabilities and 90% credible intervals
    for demographic categories

    Parameters
    ----------
    clf : sklearn clf
        fitted clf with predict object
    df : dataframe
        containing labeled columns to test for bias
    feature_names : list of strings
        names of features used in fitting clf
    categories : list of strings
        demographic columns in df to test for bias, e.g. ['gender', ...]
    low : float
        lower bound threshold
    high : float
        upper bound threshold
    num : int
        number of values in threshold range

    Returns
    -------
    thresholds_to_check : np.array
        np.linspace(low, high, num)
        range of float thresholds checked for posterior prob.
    post_probs : defaultdict(list)
        containing posterior probability at each threshold

    Example
    ---------
    given
    1) a classifier, clf, trained on:
        three features ('feat1', 'feat2', 'feat3')
        and a target 'y', containing values ranging from 0 to 1, and
    2) a df containing columns: ['feat1', 'feat2', 'feat3', 'gender']

    classifier_posterior_probabilities(clf,
                                        df,
                                        ['feat1', 'feat2', 'feat3'],
                                        ['gender'])
    >>> np.array([0.6, ..., 0.9]), defaultdict(list, {'gender__F':[.1,...]})
    """

    # get decision score for each user and sort by the score

    # this sort makes finding who matches at a threshold easy
    X = df[feature_names].values

    # copy dataframe and add decision column:
    df = df.copy()
    # require clf to have decision_function or predict_proba
    clf = ClassifierWrapper(clf)
    df['decision'] = clf.decision_function(X)
    # allow for older and newer pandas sorting schemes
    if hasattr(df, 'sort_values'):
        sorted_df = df.reindex(
            df.sort_values(
                'decision',
                ascending=False).index)
    else:
        sorted_df = df.reindex(df.sort('decision', ascending=False).index)

    n_samples = sorted_df.shape[0]

    # define range of values to test over if not inputted
    if low is None:
        low = df.decision.min()
    if high is None:
        high = df.decision.max()

    thresholds_to_check = np.linspace(low, high, num)
    post_probs = defaultdict(list)

    for thresh in thresholds_to_check:
        # set the top 1-thresh proportion of sample to 1 (match) and the
        # rest to 0 (not match)
        num_matches = int(n_samples * (1 - thresh))
        num_not_matches = (n_samples - int(n_samples * (1 - thresh)))
        matched_col = get_unique_name('matched', df.columns)
        sorted_df[matched_col] = ([1] * num_matches) + ([0] * num_not_matches)

        # bias category probabilities
        for category in categories:
            cat_vals = set(sorted_df[category].dropna())
            for val in cat_vals:
                n = sum(sorted_df[category] == val)
                x = sum(np.logical_and(sorted_df[category] == val,
                                       sorted_df[matched_col]))
                a = x + 0.5
                b = n - x + 0.5
                l, u = stats.beta.interval(0.95, a, b, loc=0, scale=1)
                feat_sim = [
                    round(stats.beta.mean(a, b, loc=0, scale=1), 3),
                    round(l, 3),
                    round(u, 3)
                ]
                post_probs[str(category) + '__' + str(val)].append(feat_sim)

    return thresholds_to_check, post_probs


def get_bias_chi2_pvals(clf, df, feature_names, categories,
                        low=None, high=None, num=100):
    """
    Get p-values across a range of decision thresholds

    Parameters
    ------------
    clf : sklearn clf object
        model classifier, must have a `decision_function` or
        `predict_proba` method
    df : pandas DataFrame
        contains untransformed data
    feature_names : list of strings
        features included in the classifier
    categories : list of strings
        names of demographic columns to check, e.g. ['gender', 'ethnicity']
    low : float
        lower threshold value
    high : float
        upper threshold value
    num : int
        number of thresholds to consider

    Returns
    ---------
    thresholds_to_check : range of floats
        decision thresholds obtained by np.linspace(low, high,num)
    post_chi2stat_pvals : defaultdict(list)
        containing categories' chi2 statistics and p_vals at a range
        of thresholds

    """

    # get decision score for each user and sort by the score
    # this sort makes finding who matches at a threshold easy
    X = df[feature_names].values

    # subsequent modifications on copy of the input dataframe
    df = df.copy()
    clf = ClassifierWrapper(clf)
    df['decision'] = clf.decision_function(X)
    # allow for older and newer pandas sorting schemes
    if hasattr(df, 'sort_values'):
        sorted_df = df.reindex(
            df.sort_values(
                'decision',
                ascending=False).index)
    else:
        sorted_df = df.reindex(df.sort('decision', ascending=False).index)

    matched_col = get_unique_name('matched', df.columns)

    # define range of values to test over if not inputted
    if low is None:
        low = df.decision.min()
    if high is None:
        high = df.decision.max()

    n_samples = sorted_df.shape[0]
    thresholds_to_check = np.linspace(low, high, num)
    post_chi2stat_pvals = defaultdict(list)

    for threshold in thresholds_to_check:
        # set the top 1-threshold proportion of sample to 1 (match) and the
        # rest to 0 (not match)
        num_matches = int(n_samples * (1 - threshold))
        num_not_matches = (n_samples - int(n_samples * (1 - threshold)))
        sorted_df[matched_col] = ([1] * num_matches) + ([0] * num_not_matches)

        for category in categories:
            # get p-values for non-nan values
            category_vals = set(sorted_df[category].dropna())
            cat_df = sorted_df[sorted_df[category].isin(category_vals)]
            cat_ctabs = pd.crosstab(cat_df[matched_col], cat_df[category])
            chi2_stat, chi2_pval = chi2_contingency(cat_ctabs)[:2]
            post_chi2stat_pvals[category].append((chi2_stat, chi2_pval))

    return thresholds_to_check, post_chi2stat_pvals


def generate_bayesfactors(clf, df, feature_names, categories,
                          prior_strength='', hyperparam=(1, 1, 1, 1),
                          N=1000, low=None, high=None, num=100):
    """
    Function to check demographic bias of clf with reference to
    dataframe containing labeled features. Decision functions for test
    data are sorted, and users in top thresholds, in terms of decision
    functions, are considered 'matched'. Proportion of matched users across
    demographic categories are used to estimate posterior densities of
    probability of being matched within demographic categories. The mean and
    90% credible intervals of ratios of probabilities between categories are
    calculated and can be plotted using `viz.get_bias_plots`.
    The intervals are compared to the null hypothetical bounds of
    0.8-1.2, with reference to the "4/5ths Rule" for adverse impact,
    across the range of thresholds. If the interval overlaps the
    bounds across thresholds, then the clf can be considered to be 'unbiased'.

    Parameters
    ----------

    clf : sklearn clf
        fitted clf with predict object
    df : pandas DataFrame
        reference dataframe containing labeled features to test for bias
    feature_names : list of strings
        names of features used in fitting clf
    categories : list of strings
        names of categories to test for bias, e.g. ['gender', ...]
    prior_strength : string from {'weak', 'strong', 'uniform'}
        prior distribution to be 'informative'/'noninformative'/'uniform'
    hyperparam : tuple (alpha1, beta1, alpha2, beta2)
        optional manual input of hyperparameters
    N : int
        number of posterior samples to draw for each simulation
    low, high, num : float, float, int
        range of values for thresholds

    Returns
    -------
    thresholds_to_check : array of floats
        decision thresholds obtained by np.linspace(low, high, num)
    ratio_probs : OrderedDict()
        contains ratios of matched probabilities within demographic categories
    N : int
        number of posterior samples drawn for each simulation
    """

    X = df[feature_names].values

    df = df.copy()
    # get decision score for each user and sort by the score
    clf = ClassifierWrapper(clf)
    df['decision'] = clf.decision_function(X)
    # this sort makes finding who matches at a threshold easy
    # allow for older and newer pandas sorting schemes
    if hasattr(df, 'sort_values'):
        sorted_df = df.reindex(
            df.sort_values(
                'decision',
                ascending=False).index)
    else:
        sorted_df = df.reindex(df.sort('decision', ascending=False).index)

    # define range of values to test over if not inputted
    if low is None:
        low = df.decision.min()
    if high is None:
        high = df.decision.max()

    n_samples = sorted_df.shape[0]
    thresholds_to_check = np.linspace(low, high, num)

    ratio_probs = defaultdict(list)

    for thresh in thresholds_to_check:
        # set the top (1-thresh) of samples to 1 (match), rest to 0 (not match)
        num_matches = int(n_samples * (1 - thresh))
        num_not_matches = (n_samples - int(n_samples * (1 - thresh)))
        matched_col = get_unique_name('matched', df.columns)
        sorted_df[matched_col] = ([1] * num_matches) + ([0] * num_not_matches)

        # for all categories, generate contigency tables
        # and simulate posterior sample for ratio of matched probabilities
        for category in categories:
            feature_vals = set(df[category].dropna())
            for v1, v2 in combinations(feature_vals, 2):

                mask = np.logical_or(df[category] == v1,
                                     df[category] == v2)

                ct_table = pd.crosstab(
                    sorted_df.loc[mask, category],
                    sorted_df.loc[mask, matched_col]
                ).values

                feat_sim = sim_beta_ratio(ct_table, thresh,
                                          prior_strength, hyperparam, N)
                if v1 > v2:
                    out_string = "%s over %s" % (str(v2), str(v1))
                    ratio_probs[out_string] = OrderedDict()
                    ratio_probs[out_string][thresh] = feat_sim

                else:
                    out_string = "%s over %s" % (str(v1), str(v2))
                    ratio_probs[out_string] = OrderedDict()
                    ratio_probs[out_string][thresh] = feat_sim

    return thresholds_to_check, ratio_probs, N


def sim_beta_ratio(table, threshold, prior_strength, hyperparam, N):
    """
    Calculates simulated ratios of match probabilites using a beta
    distribution and returns corresponding means and 95% credible
    intervals, posterior parameters, Bayes factor

    Parameters
    ------------
    table : 2x2 numpy array
        corresponds to contingency table,
        for example,
               False    True
        GroupA   5       4
        GroupB   3       4
        contains frequency counts: [[5, 4], [3, 4]]
    threshold : float
        value to split continuous variable on
    prior_strength : string from {'weak', 'strong', 'uniform'}
        prior distribution to be 'informative'/'noninformative'/'uniform'
    N : int
        number of posterior samples to draw for each simulation

    Returns
    ------------
    list : means and 95% credible intervals, posterior parameters, Bayes factor
    """

    n_sim = N
    # store array of total counts in table by category
    category_counts = table.sum(axis=1, dtype=float)
    # store array of number of matches by categories
    match_counts = table[:, 1]
    # set hyperparameters according to threshold and sample size
    if prior_strength == 'weak':
        # weakly informative prior, has standard deviation
        # of 0.1 at alpha / (alpha + beta) = 0.5
        # coefficient 24 is empirically derived for best smoothing at small N
        alpha1, beta1 = (1 - threshold) * 24., threshold * 24.
        alpha2, beta2 = (1 - threshold) * 24., threshold * 24.
    elif prior_strength == 'strong':
        # observing 'idealized' dataset of size n
        alpha1 = round((1 - threshold) * category_counts[0])
        beta1 = round(threshold * category_counts[0])
        alpha2 = round((1 - threshold) * category_counts[1])
        beta2 = round(threshold * category_counts[1])
    elif prior_strength == 'uniform':
        # uniform prior
        alpha1, beta1 = 1, 1
        alpha2, beta2 = 1, 1
    else:
        # user specified, defaults to uniform
        alpha1, beta1, alpha2, beta2 = hyperparam

    # draw posterior sample of matching probabilities
    post_alpha1 = alpha1 + match_counts[0]
    post_beta1 = beta1 + category_counts[0] - match_counts[0]

    post_alpha2 = alpha2 + match_counts[1]
    post_beta2 = beta2 + category_counts[1] - match_counts[1]

    p1 = np.random.beta(post_alpha1, post_beta1, n_sim)
    p2 = np.random.beta(post_alpha2, post_beta2, n_sim)

    # posterior draw of ratios
    p1p2 = p1 / p2
    p2p1 = p2 / p1

    # For fraction of posterior ratios in range [.8, 1.25], get Bayes factor
    post_prob_null = np.sum((p1p2 >= 0.8) & (p1p2 <= 1.25)) / float(n_sim)
    bayes_factor = post_prob_null / (1 - post_prob_null)

    return [np.mean(p1p2), np.mean(p2p1), np.std(p1p2), np.std(p2p1),
            np.percentile(p1p2, 2.5), np.percentile(p2p1, 2.5),
            np.percentile(p1p2, 97.5), np.percentile(p2p1, 97.5),
            (post_alpha1, post_beta1), (post_alpha2, post_beta2), bayes_factor]
