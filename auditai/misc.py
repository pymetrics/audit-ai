from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, f_oneway
from sklearn.metrics import mutual_info_score

from .simulations import classifier_posterior_probabilities
from .utils.crosstabs import (crosstab_bayes_factor,
                              crosstab_ztest,
                              top_bottom_crosstab)
from .utils.validate import boolean_array, check_consistent_length


def default_thresh_value(results):
    if len(set(results)) == 2:
        return np.mean(list(set(results)))
    return np.median(results)


def anova(labels, results, subset_labels=None):
    """
    Returns one-way ANOVA f-statistic and p-value from
    input vectors of categorical labels and numeric results

    Parameters
    ------------
    labels : array_like
        containing categorical values like ['M', 'F']
    results : array_like
        containing real numbers
    subset_labels : list of strings, optional
        if only specific labels should be included

    Returns
    ----------
    F_onewayResult : scipy.stats object (essentially a 2-tuple)
        contains one-way f-statistic and p-value, indicating whether
        scores have same sample mean

    """
    check_consistent_length(labels, results)

    df = pd.DataFrame(list(zip(labels, results)), columns=['label', 'result'])
    if subset_labels is not None:
        df = df.loc[df['label'].isin(subset_labels)]

    unique_labels = df['label'].dropna().unique()
    score_vectors = [df.loc[df['label'] == lab, 'result']
                     for lab in unique_labels]
    return f_oneway(*score_vectors)


def bias_test_check(labels, results, category=None, test_thresh=None,
                    **kwargs):
    """
    Utility function for checking if statistical tests are passed
    at a reference threshold

    Parameters
    --------
    labels : array_like
        containing categorical values like ['M', 'F']
    results : array_like
        containing real numbers
    category : string, optional
        the name of the category labels are in, e.g. 'Gender'
    test_thresh : numeric
        threshold value to test
    **kwargs : optional additional arguments for compare_groups

    Returns
    --------
    print statement indicating whether specific statistical tests pass or fail
    """
    if test_thresh is None:
        test_thresh = default_thresh_value(results)

    min_props, z_ps, fisher_ps, chi_ps, bfs = compare_groups(
        labels, results, low=test_thresh, num=1, **kwargs)

    # if no category is specified, concatenate strings
    if category is None:
        category = '_vs_'.join([str(i) for i in set(labels)])[:20]
    # test if passes at test_thresh
    bias_tests = {'4/5': {'results': min_props,
                          'check': lambda x: x < 0.8},
                  'Fisher exact': {'results': fisher_ps,
                                   'check': lambda x: x < 0.05},
                  'Chi squared': {'results': chi_ps,
                                  'check': lambda x: x < 0.05},
                  'z': {'results': z_ps,
                        'check': lambda x: x < 0.05},
                  'Bayes Factor': {'results': bfs,
                                   'check': lambda x: x > 3.}
                  }

    for name, test in bias_tests.items():
        stat_value = test['results'].get(test_thresh)
        if stat_value and not test['check'](stat_value):
            print('*{} passes {} test at {:.2f}*'.format(
                category, name, test_thresh))
        elif stat_value is not None:
            print('*{} fails {} test at {:.2f}*'.format(
                category, name, test_thresh))
            print(" - {} minimum proportion at {:.2f}: {:.3f}".format(
                category, test_thresh, stat_value))
        else:
            print("Unable to run {} test".format(name))


def make_bias_report(clf, df, feature_names, categories, **kwargs):
    """
    Utility function for report dictionary from
    `classifier_posterior_probabilities`. Used for plotting
    bar plots in `bias_bar_plot`

    Parameters
    -----------
    clf : sklearn clf
        fitted clf with predict object
    df : pandas DataFrame
        reference dataframe containing labeled features to test for bias
    feature_names : list of strings
        names of features used in fitting clf
    categories : list of strings
        names of categories to test for bias, e.g. ['gender']
    ref_threshold : float (optional)
        cutoff value at which to generate metrics
    **kwargs : optional additional arguments for
        classifier_posterior_probabilities, specifically low, high, num

    Returns
    --------
    out_dict : dictionary
        contains category names, average probabilities and errors by category
        of form {'gender': {'categories':['F', 'M'],
                            'averages': [.5, .5],
                            'errors': [.1, .1]}
                }
    """
    ref_threshold = kwargs.pop("ref_threshold")
    threshes, probs = classifier_posterior_probabilities(
        df, clf, feature_names, categories, **kwargs)

    # if not specified, set ref_threshold at 80% of max(threshes)
    if ref_threshold is None:
        idx_80 = int(len(threshes)*.8)
        ref_threshold = sorted(threshes)[idx_80]

    ref_idx = list(threshes).index(ref_threshold)

    out_dict = {}
    for category in categories:
        cat_vals = [k.split('__')[1]
                    for k in probs.keys() if k.split('__')[0] == category]
        cat_avgs = [probs[val][ref_idx][0] for val in cat_vals]
        cat_errors = [probs[val][ref_idx][1:] for val in cat_vals]
        out_dict[category] = {
            'categories': cat_vals,
            'averages': cat_avgs,
            'errors': cat_errors}

    return out_dict


def get_group_proportions(labels, results, **kwargs):
    """
    Returns pass proportions for each group present in labels, according to
    their results

    Parameters
    ------------
    labels : array_like
        contains categorical labels
    results : array_like
        contains numeric or boolean values
    **kwargs : optional
        additional values for thresholds to test:
        low : float
            if None, will default to min(results)
        high : float
            if None, will default to max(results)
        num : int, default 100
            number of thresholds to check

    Returns
    --------
    prop_dict: dictionary
        contains {group_name : [[thresholds, pass_proportions]]}

    """
    low = kwargs.get("low", min(results))
    high = kwargs.get("high", max(results))
    num = kwargs.get("num", 100)
    thresholds = np.linspace(low, high, num).tolist()
    groups = set(labels)
    prop_dict = defaultdict(list)

    for group in groups:
        pass_props = []
        for thresh in thresholds:
            decs = [i <= thresh for i in results]
            crosstab = pd.crosstab(pd.Series(labels), pd.Series(decs))
            row = crosstab.loc[group]
            pass_prop = row[True] / float(row.sum())
            pass_props.append(pass_prop)
        prop_dict[group].append(thresholds)
        prop_dict[group].append(pass_props)
    return prop_dict


def compare_groups(labels, results,
                   low=None, high=None, num=100,
                   comp_groups=None, print_skips=False):
    """
    Function to plot proportion of largest and smallest bias groups and
    get relative z scores

    Parameters
    --------
    labels : array_like
        contains categorical values like ['M', 'F']
    results : array_like
        contains real numbers, e.g. threshold scores or floats in (0,1)
    low : float
        lower threshold value
    high : float
        upper threshold value
    num : int
        number of thresholds to check
    comp_groups : list of strings, optional
        subset of labels to compare, e.g. ['white', 'black']
    print_skips : bool
        whether to display thresholds skipped

    Returns
    ---------
    min_props : dict
        contains (key, value) of (threshold : max group/min group proportions)
    z_ps : dict
        contains (key, value) of (threshold : p-value of two tailed z test)
    fisher_ps : dict
        contains (key, value) of (threshold : p-value of fisher exact test)
    chi_ps : dict
        contains (key, value) of (threshold : p-value of chi squared test)
    bayes_facts : dict
        contains (key, value) of (threshold : bayes factor)
    """

    # cast labels and scores to pandas Series
    df = pd.DataFrame(list(zip(labels, results)), columns=['label', 'result'])

    min_props = {}
    fisher_ps = {}
    chi_ps = {}
    z_ps = {}
    bayes_facts = {}

    if comp_groups is not None:
        df = df[df['label'].isin(comp_groups)]

    # define range of values to test over if not inputted
    if low is None:
        low = min(results)
    if high is None:
        high = max(results)

    thresholds = np.linspace(low, high, num)

    skip_thresholds = []
    for thresh in thresholds:

        df['dec'] = [i >= thresh for i in results]

        # compare rates of passing across groups
        ctabs = pd.crosstab(df['label'], df['dec'])

        # skip any thresholds for which the crosstabs are one-dimensional
        if 1 in ctabs.shape:
            skip_thresholds.append(thresh)
            continue

        normed_ctabs = ctabs.div(ctabs.sum(axis=1), axis=0)
        true_val = max(set(df['dec']))
        max_group = normed_ctabs[true_val].max()
        normed_proportions = normed_ctabs[true_val] / max_group
        min_proportion = normed_proportions.min()

        # run statistical tests
        if ctabs.shape == (2, 2):
            test_results = test_multiple(df['label'].values, df['dec'].values)
            z_pval = test_results.get('z_score')[1]
            fisher_pval = test_results.get('fisher_p')[1]
            chi2_pval = test_results.get('chi2_p')[1]
            bayes_fact = test_results.get('BF')

        else:
            top_bottom_ctabs = top_bottom_crosstab(df['label'], df['dec'])
            z_pval = crosstab_ztest(top_bottom_ctabs)[1]
            fisher_pval = fisher_exact(top_bottom_ctabs)[1]
            chi2_pval = chi2_contingency(ctabs)[1]
            bayes_fact = crosstab_bayes_factor(ctabs)

        min_props[thresh] = min_proportion
        z_ps[thresh] = z_pval
        fisher_ps[thresh] = fisher_pval
        chi_ps[thresh] = chi2_pval
        bayes_facts[thresh] = bayes_fact

    if len(skip_thresholds) > 0 and print_skips:
        print('One-dimensional thresholds were skipped: {}'.format(
            skip_thresholds))
    return min_props, z_ps, fisher_ps, chi_ps, bayes_facts


def proportion_test(labels, decisions):
    """
    Compare rates of passing across groups,
    relative to the highest passing group

    Parameters
    ----------
    labels : array_like
        categorical labels for each corresponding value of `decision` ie. M/F

    decisions : array_like
        binary decision values, ie. True/False, 0/1 or 'pass'/'fail'
        NB: the 'passing' value must evaluate to greater than the failing value

    Returns
    -------
    normed_proportions : pd.Series
        displays pass rates by `label` group
        relative to the highest passing group (which itself is always 1.0)
    """
    check_consistent_length(labels, decisions)
    decisions = boolean_array(decisions)
    crosstab = pd.crosstab(pd.Series(labels), pd.Series(decisions))

    # require crosstab not to be one-dimensional (e.g. one kind of label)
    if 1 in crosstab.shape:
        raise ValueError('One-dimensional data has no proportions')

    normed_ctabs = crosstab.div(crosstab.sum(axis=1), axis=0)
    true_val = max(set(decisions))
    max_group = normed_ctabs[true_val].max()
    normed_proportions = normed_ctabs[true_val] / max_group
    return normed_proportions


def test_multiple(labels, decisions,
                  tests=('ztest', 'fisher', 'chi2', 'BF', 'prop'),
                  display=False):
    """
    Function that returns p_values for z-score, fisher exact, and chi2 test
    of 2x2 crosstab of passing rate by labels and decisions

    See docs for z_test_ctabs, fisher_exact, chi2_contingency and
    bf_ctabs for details of specific tests

    Parameters
    ----------
    labels : array_like
        categorical labels for each corresponding value of `decision` ie. M/F

    decisions : array_like
        binary decision values, ie. True/False or 0/1

    tests : list
        a list of strings specifying the tests to run, valid options
        are 'ztest', 'fisher', 'chi2' and 'bayes'. Defaults to all four.
        -ztest: p-value for two-sided z-score for proportions
        -fisher: p-value for Fisher's exact test for proportions
        -chi2: p-value for chi-squared test of independence for proportions
        -bayes: bayes factor for independence assuming uniform prior
        -prop: proportion of lowest to highest passing rates by group

    display : bool
        print the results of each test in addition to returning them

    Returns
    -------
    results : dict
        dictionary of values, one for each test.
        Valid keys are: 'z_score', 'fisher_p', 'chi2_p', 'BF', and 'prop'

    Examples
    --------
    >>> # no real difference between groups
    >>> labels = ['group1']*100 + ['group2']*100 + ['group3']*100
    >>> decisions = [1,0,0]*100
    >>> all_test_ctabs(dependent_ctabs)
        (0.0, 1.0, 1.0, 0.26162148804907587)

    >>> # massively biased ratio of hits/misses by group
    >>> ind_ctabs = np.array([[75,50],[25,50]])
    >>> all_test_ctabs(ind_ctabs)
        (-3.651483716701106,
         0.0004203304586999487,
         0.0004558800052056139,
         202.95548692414306)

    >>> # correcting with a biased prior
    >>> biased_prior =  np.array([[5,10],[70,10]])
    >>> all_test_ctabs(ind_ctabs, biased_prior)
        (-3.651483716701106,
         0.0004203304586999487,
         0.0004558800052056139,
         0.00012159518854984268)
    """

    decisions = boolean_array(decisions)
    crosstab = pd.crosstab(pd.Series(labels), pd.Series(decisions))
    crosstab = crosstab.values

    # can only perform 2-group z-tests & fisher tests
    # getting crosstabs for groups with highest and lowest pass rates
    # as any difference between groups is considered biased
    tb_crosstab = top_bottom_crosstab(labels, decisions)

    results = {}
    if 'ztest' in tests:
        results['z_score'] = crosstab_ztest(tb_crosstab)
    if 'fisher' in tests:
        # although fisher's exact can be generalized to multiple groups
        # scipy is limited to shape (2, 2)
        # TODO make generalized fisher's exact test
        # returns oddsratio and p-value
        results['fisher_p'] = fisher_exact(tb_crosstab)[:2]
    if 'chi2' in tests:
        # returns chi2 test statistic and p-value
        results['chi2_p'] = chi2_contingency(crosstab)[:2]
    if 'BF' in tests:
        results['BF'] = crosstab_bayes_factor(crosstab)
    if 'prop' in tests:
        results['prop'] = min(proportion_test(labels, decisions))

    if display:
        for key in results:
            print("{}: {}".format(key, results[key]))

    return results


def quick_bias_check(clf, df, feature_names, categories, thresh_pct=80,
                     pass_ratio=.8):
    """
    Useful for generating a bias_report more quickly than make_bias_report
    simply uses np.percentile for checks

    Parameters
    -----------
    clf : sklearn clf
        fitted clf with predict object
    df : pandas DataFrame
        reference dataframe containing labeled features to test for bias
    feature_names : list of strings
        names of features used in fitting clf
    categories : list of strings
        names of categories to test for bias, e.g. ['gender', 'ethnicity']
    thresh_pct : float, default 80
        percentile in [0, 100] at which to check for pass rates
    pass_ratio : float, default .8
        cutoff specifying whether ratio of min/max pass rates is acceptable

    Returns
    --------
    passed: bool
        indicates whether all groups have min/max pass rates >= `pass_ratio`
    bias_report : dict
        of form {'gender': {'categories':['F', 'M'],
                            'averages': [.2, .22],
                            'errors': [[.2, .2], [.22, .22]]}
                }
    min_bias_ratio : float
        min of min_max_ratios across all categories
        if this value is less than `pass_ratio`, passed == False
    """

    bdf = df.copy()
    X = bdf.loc[:, feature_names].values
    decs = clf.decision_function(X)
    bdf['score'] = decs

    min_max_ratios = []
    bias_report = {}
    for category in categories:
        cat_df = bdf[bdf[category].notnull()]
        cat_df['pass'] = cat_df.score > np.percentile(cat_df.score, thresh_pct)
        cat_group = bdf.groupby(category).mean()['pass']
        cat_dict = cat_group.to_dict()
        min_max_ratios.append(cat_group.min()/float(cat_group.max()))
        bias_report[category] = {'averages': cat_dict.values(),
                                 'categories': cat_dict.keys(),
                                 'errors': [[i, i] for i in cat_dict.values()]
                                 }

    passed = all(np.array(min_max_ratios) >= pass_ratio)
    min_bias_ratio = min(min_max_ratios)
    return passed, bias_report, min_bias_ratio


def one_way_mi(df, feature_list, group_column, y_var, bins):

    """
    Calculates one-way mutual information group variable and a
    target variable (y) given a feature list regarding.

    Parameters
    ----------
    df : pandas DataFrame
         df with features used to train model, plus a target variable
         and a group column.
    feature_list : list DataFrame
        List of strings, feature names.
    group_column : string
        name of column for testing bias, should contain numeric categories
    y_var : string
        name of target variable column
    bins : tuple
        number of bins for each dimension

    Returns
    -------
    mi_table : pandas DataFrame
        data frame with mutual information values, with one row per feature
        in the feature_list, columns for group and y.
    """

    group_cats = df[group_column].values
    y_cats = df[y_var].values

    c_g = [
        np.histogramdd([np.array(df[feature]), group_cats], bins=bins)[0]
        for feature in feature_list
        ]
    c_y = [
        np.histogramdd([np.array(df[feature]), y_cats], bins=bins)[0]
        for feature in feature_list
        ]

    # compute mutual information (MI) between trait and gender/eth/y
    mi_g = [mutual_info_score(None, None, contingency=i) for i in c_g]
    mi_y = [mutual_info_score(None, None, contingency=i) for i in c_y]
    mi_table = pd.DataFrame({'feature': feature_list,
                             group_column: mi_g,
                             y_var: mi_y})

    # NOTE: Scale group and y where the highest MI is scaled to 1 to
    # facilitate interpreting relative importance to bias and performance
    mi_table["{}_scaled".format(group_column)] = (
        mi_table[group_column] / mi_table[group_column].max()
    )
    mi_table["{}_scaled".format(y_var)] = (
        mi_table[y_var] / mi_table[y_var].max()
    )

    return mi_table
