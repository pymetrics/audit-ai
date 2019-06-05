import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .misc import (compare_groups,
                   get_group_proportions,
                   make_bias_report)
from .simulations import (get_bias_chi2_pvals,
                          generate_bayesfactors)

plt.style.use('fivethirtyeight')


def plot_bias_pvals(thresholds, pvals, category, sig_thresh=.05, outpath=None):
    """
    Plots Chi-Squared p-value test of match rate differences
    within demographic groups. Uses outputs of `get_bias_chi2_pvals`

    Parameters
    ----------
    thresholds : array_like, range of floats
        decision thresholds obtained by np.linspace(low,high,num)
    pvals : array_like, range of floats
        p-values from Chi-Squared test
    category : string
        demographic category to test for bias, e.g. 'Gender'
    sig_thresh : float
        y-axis value to draw threshold line
    outpath : string, optional
        location in memory to save plot

    Returns
    -------
    None

    Plots
    ------
    plot with Chi-Squared p-values
    """

    low = np.floor(min(thresholds))
    high = np.ceil(max(thresholds))

    plot_title = 'Bias Chi-Squared p-val tests for %s' % (category)

    plt.figure()
    plt.scatter(thresholds, pvals)
    plt.axis([low, high, -.1, 1.1])
    plt.title(plot_title)
    plt.ylabel('p-val')
    plt.xlabel('Threshold')
    plt.axhline(y=sig_thresh, ls='dashed', color='r')
    if outpath is not None:
        plt.savefig(outpath)
    plt.show()


def plot_bias_test(thresholds, ratios, category, outpath=None):
    """
    Plots prediction ratio for different groups across thresholds
    Checks 4/5ths test -- i.e. whether any group passes 20+% more than another

    Parameters
    ----------
    thresholds : array_like
        range of floats - thresholds obtained by np.linspace(low,high,num)
    ratios : dict
        output of `generate_bayesfactors`, contains means, lowers, uppers
    category : string
        demographic category to test for bias, e.g. 'Gender'
    outpath : string, optional
        location in memory to save plot

    Returns
    -------
    None

    Plots
    ------
    plot with prediction ratios for different groups
    """

    means = [i[0] for i in ratios.values()]
    lowers = [i[4] for i in ratios.values()]
    uppers = [i[6] for i in ratios.values()]

    low = np.floor(min(thresholds))
    high = np.ceil(max(thresholds))

    plot_title = 'Bias Tests for %s' % (category)

    plt.figure()
    plt.scatter(thresholds, means)
    plt.plot(thresholds, lowers, '-', color='g')
    plt.plot(thresholds, uppers, '-', color='g')
    plt.axis([low, high, 0, max(2, max(uppers) + 0.1)])
    plt.title(plot_title)
    plt.ylabel('Ratio')
    plt.xlabel('Threshold')
    plt.axhline(y=0.8, ls='dashed', color='r')
    plt.axhline(y=1.25, ls='dashed', color='r')
    if outpath is not None:
        plt.savefig(outpath)
    plt.show()


def get_bias_plots(clf, df, feature_names, categories, **kwargs):
    """
    Generate bias plots from a classifier

    Parameters
    ------------
    clf : sklearn fitted clf
        with predict object
    df : pandas DataFrame
        untransformed df
    feature_names : list of strings
        names of features used in fitting clf
    categories : list of strings
        demographic column names used to test for bias,
        e.g. ['gender', 'ethnicity']
    **kwargs :
        additional arguments for `generate_bayes_factors` and
        `get_bias_chi2_pvals`, such as `low`, `high`, `num`, `prior_strength`

    Returns
    ------------
    None

    Plots
    -------
    Linear plots of simulated pass ratios and chi squared p-values
    at assorted thresholds
    """

    thresholds, ratios, N = generate_bayesfactors(
        clf, df, feature_names, categories, **kwargs
    )
    keys = sorted(ratios.keys())
    for key in keys:
        plot_bias_test(thresholds, ratios[key], key)

    chi2_thresholds, chi2stat_pvals = get_bias_chi2_pvals(
        clf, df, feature_names, categories, **kwargs
    )
    for category in categories:
        chi2_pvals = [i[1] for i in chi2stat_pvals[category]]
        plot_bias_pvals(chi2_thresholds, chi2_pvals, category)


def bias_bar_plot(clf=None, df=None, feature_names=None,
                  categories=None, low=.01, high=.99, num=99,
                  ref_threshold=None, bias_report=None):
    """
    Plot bar plots for overall recommendation by bias group
    NB: if bias_report is not None, only ref_threshold is also needed

    Parameters
    ------------
    clf : sklearn clf
        fitted clf with predict object
    df : pandas DataFrame
        reference dataframe containing labeled features to test for bias
    feature_names : list of strings
        names of features used in fitting clf
    categories : list of strings
        names of categories to test for bias, e.g. ['gender']
    low, high, num : float, float, int
        range of values for thresholds
    ref_threshold : float
        cutoff value at which to generate plot
    bias_report: dict, optional
        output of make_bias_report

    Returns
    ------------
    ax : matplotlib subplot axes

    Plots
    --------
    Bar plot containing recommendations across each bias group
    """
    if all(kw is None for kw in (clf, df, feature_names, categories,
                                 bias_report)):
        raise ValueError('if bias_report is None, other arguments cannot be!')

    if ref_threshold is None:
        raise ValueError('ref_threshold must be defined')

    if bias_report is None:
        bias_report = make_bias_report(clf, df, feature_names, categories,
                                       low, high, num, ref_threshold)

    for key in bias_report:
        data = bias_report[key]
        errs = map(lambda i: (i[1] - i[0]) / 2., data['errors'])
        labels = data['categories']
        avgs = data['averages']
        ind = np.arange(len(labels))

        fig, ax = plt.subplots()
        ax.bar(ind, avgs, ref_threshold, yerr=errs)
        ax.set_title(key)
        ax.set_xticks(ind + ref_threshold // 2)
        ax.set_xticklabels(labels)
        plt.show()

    return ax


def plot_threshold_tests(labels, results, category=None, comp_groups=None,
                         include_metrics=('Proportions',
                                          'z-test p-vals',
                                          'Fisher Exact p-vals',
                                          'Chi Squared p-vals',
                                          'Bayes factors'),
                         ref_prop=0.8, ref_z=0.05, ref_fisher_p=0.05,
                         ref_chi2_p=0.05, ref_bayes=3., **kwargs):
    """
    Plot results of following tests across thresholds:
    - max-vs-min pass proportions
    - two-tailed z-test for max-vs-min groups
    - Fisher exact test for max-vs-min groups
    - Chi Squared test of independence for all groups
    - Bayes factor

    Parameters
    --------
    labels : array_like
        containing categorical values like ['M', 'F']
    results : array_like
        containing real numbers
    category : string
        corresponding to category of labels, e.g. 'Gender'
    comp_groups : list of strings
        optional -- subset of 2 labels to include, e.g. ['M', 'F']
    include_metrics : list/tuple of strings
        from `{'Proportions', 'z-scores', 'Fisher Exact p-vals',
               'Chi Squared p-vals', 'Bayes factors'}`
        NB : if only one metric plot is wanted, use a trailing comma,
             e.g. `include_metrics = ('Proportions', )`
    ref_prop, ref_z, ref_fisher_p, ref_chi2_p, ref_bayes : floats
        designated values to plot horizontal reference line for
        corresponding metric
    **kwargs : additional arguments for continuous.compare_groups,
        e.g. low, high, num for defining threshold values

    Returns
    --------
    axarr : numpy.ndarray
        contains matplotlib subplot axes

    Plots
    --------
    Values for desired statistical tests across threshold values
    """

    reference_vals = (ref_prop, ref_z, ref_fisher_p, ref_chi2_p, ref_bayes)

    # supported metrics
    metric_names = (
        'Proportions',
        'z-test p-vals',
        'Fisher Exact p-vals',
        'Chi Squared p-vals',
        'Bayes factors')

    # check to make sure only allowed metrics are included:
    if not set(include_metrics).issubset(set(metric_names)):
        raise KeyError(
            "include_metrics must be within "
            "{'Proportions', 'z-test p-vals', 'Fisher Exact p-vals', "
            "'Chi Squared p-vals', 'Bayes factors'}")
    # tuple of dictionaries containing {threshold: metric_value} pairs
    metric_dicts = compare_groups(labels, results, **kwargs)
    # map those dictionaries to their names
    names_mets_refvals_dict = dict(
        zip(metric_names, zip(metric_dicts, reference_vals)))

    # create subplots
    num_subplots = len(include_metrics)
    labels = list(map(str, labels))

    fig, axarr = plt.subplots(
        num_subplots, sharex=True, figsize=(
            12, 3 + 2 * num_subplots))

    if category is None:
        category = ' vs '.join(sorted(set(labels)))

    fig.suptitle('Model Statistical Bias Tests %s' % category, fontsize=16)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    group_name = 'Min-Max Group'

    if comp_groups is not None:
        group_name = '%s vs %s' % (comp_groups[:2])

    def _make_metric_plot(ax, input_dict, metric_name, group_name,
                          metric_bound=.05):
        """internal function for making a standalone metrics plot"""
        x_vals, y_vals = zip(*sorted(input_dict.items()))
        ax.plot(x_vals, y_vals, 'o-', c='blue')
        ax.set_title('%s %s' % (group_name, metric_name), fontsize=14)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.axhline(metric_bound, c='r', linestyle='--')
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Threshold Value', fontsize=12)

    # plot relative pass rates
    def _add_metric_plot(axarr, input_dict, metric_name, group_name,
                         subplot_no=0, metric_bound=.05, last_plot=False):
        """internal function for adding a metrics subplot"""
        x_vals, y_vals = zip(*sorted(input_dict.items()))
        axarr[subplot_no].plot(x_vals, y_vals, 'o-', c='blue')
        axarr[subplot_no].set_title('%s %s' %
                                    (group_name, metric_name), fontsize=14)
        axarr[subplot_no].set_ylabel(metric_name, fontsize=12)
        axarr[subplot_no].axhline(metric_bound, c='r', linestyle='--')
        if subplot_no == 0:
            axarr[subplot_no].tick_params(labelsize=10)
        if last_plot is True:
            axarr[subplot_no].set_xlabel('Threshold Value', fontsize=12)

    for idx, met_name in enumerate(include_metrics):
        if met_name in ('Chi Squared p-vals',
                        'Bayes factors') and comp_groups is None:
            group_name = 'All Groups'
        last_plot = idx == num_subplots - 1
        met_dict, ref_val = names_mets_refvals_dict[met_name]
        if num_subplots == 1:
            _make_metric_plot(axarr, met_dict, met_name, group_name,
                              metric_bound=ref_val)
        else:
            _add_metric_plot(axarr, met_dict, met_name, group_name,
                             idx, ref_val, last_plot)

    fig.tight_layout()
    fig.subplots_adjust(top=.90)
    plt.show()

    return axarr


def plot_group_proportions(labels, results, category=None, **kwargs):
    """
    Function for plotting group pass proportions at or below various thresholds
    NB: A group whose curve lies on top of another passes less frequently
    at or below that threshold

    Parameters
    ------------
    labels : array_like
        contains categorical labels
    results : array_like
        contains numeric or boolean values
    category : string
        describes label values, e.g. 'Gender'
    **kwargs : optional
        additional values for `misc.get_group_proportions` fn
        specifically - low, high, num values for thresholds to test

    Returns
    --------
    ax : matplotlib lines object

    Plots
    -------
    single plot:
        Overlays linear plots of pass rates below range of thresholds
         results for the n groups found in labels

    """
    prop_dict = get_group_proportions(labels, results, **kwargs)
    groups = prop_dict.keys()
    for group in groups:
        x_vals, y_vals = prop_dict[group]
        ax = plt.plot(x_vals, y_vals, label=group)
    plt.legend(loc='best')
    plt.xlabel('Threshold')
    plt.ylabel('Fraction of Group Below')
    if not category:
        category = '_vs_'.join(map(str, groups))
    plt.title("%s Cumulative Pass Rate Below Threshold" % category)
    plt.show()
    return ax[0]


def plot_kdes(labels=None,
              results=None,
              category=None,
              df=None,
              label_col=None,
              result_col=None,
              colors=None,
              **kwargs):
    """
    Plots KDEs and Cumulative KDEs
    Requires seaborn for plotting

    Can either pass in arrays of labels/results or else df

    Parameters
    -----------
    labels : array_like
        categorical values
    results : array_like
        numerical values
    category : string, optional
        name of label category for plotting, e.g. 'Gender'
    df : pandas DataFrame, optional
    label_col : string, optional
        name of labels column in df
    result_col : string, optional
        name of results column in df
    colors : list of strings, optional
        takes xkcd hue labels, e.g. ['red', 'blue', 'mustard yellow']
        more here: https://xkcd.com/color/rgb/

    Returns
    --------
    ax : numpy array of matplotlib axes

    Plots
    -------
    (1,2) subplots: KDE and cumulative KDE by group in `labels`
    """
    import seaborn as sns
    if df is None:
        df = pd.DataFrame(list(zip(labels, results)),
                          columns=['label', 'result'])
    else:
        df = df.rename(columns={label_col: 'label', result_col: 'result'})
    unique_labels = df.label.dropna().unique()
    nlabels = len(unique_labels)
    if not colors:
        base_colors = ['red', 'blue']
        others = list(set(sns.xkcd_rgb.keys()) - set(base_colors))
        extra_colors = list(np.random.choice(others, nlabels, replace=False))
        colors = list(base_colors + extra_colors)[:nlabels]
    sns.set_palette(sns.xkcd_palette(colors))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    if not category:
        category = '_vs_'.join(map(str, unique_labels))
    ax[0].set_title("%s KDEs" % category)
    ax[1].set_title("%s Cumulative KDEs" % category)
    ax[0].set_ylabel('Frequency')
    ax[1].set_ylabel('Group Fraction Below')
    ax[0].set_xlabel('Threshold')
    ax[1].set_xlabel('Threshold')
    for lab in unique_labels:

        sns.kdeplot(df.loc[df.label == lab].result,
                    shade=True, label=lab, ax=ax[0], **kwargs)
        sns.kdeplot(df.loc[df.label == lab].result,
                    shade=False, label=lab, ax=ax[1],
                    cumulative=True, **kwargs)

    ax0_max_y = max([max(i.get_data()[1]) for i in ax[0].get_lines()])
    ax[0].set_ylim(0, ax0_max_y*1.1)
    plt.show()

    return ax
