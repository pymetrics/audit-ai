import numpy as np
from pandas import DataFrame
from scipy.stats import chi2, fisher_exact, chi2_contingency
from functools import partial

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
from .utils.cmh import (parse_matrix,
                        extract_data)


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


def fisher_exact_test(labels, results, threshold=None):
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

    oddsratio, pvalue = fisher_exact(ctab)
    return oddsratio, pvalue


def chi2_test(labels, results, threshold=None):
    """
    Takes list of labels and results and returns odds ratio and p-value of
    Chi-square test of independence. Uses scipy.stats.chi2_contingency,
    using an Rx2 contingency table
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

    chi2_stat, pvalue = chi2_contingency(ctab)[:2]
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


def cmh_test(dfs, pass_col=False):
    """
    The pairwise special case of the Cochran-Mantel-Haenszel test. The CMH test
    is a generalization of the McNemar Chi-Squared test of Homogenaity. Whereas
    the McNemar test examines differences over two intervals (usually before
    and after), the CMH test examines differences over any number of
    K instances.

    The Yates correction for continuity is not used; while some experts
    recommend its use even for moderately large samples (hundreds), the
    effect is to reduce the test statistic and increase the p-value. This
    is a conservative approach in experimental settings, but NOT in adverse
    impact monitoring; such an approach would systematically allow marginal
    cases of bias to go undetected.

    Parameters
    ------------
    dfs : pd.DataFrame, shape = (2,2) OR list of pd.DataFrame objects
        a K-deep list of 2x2 contingency tables
        representing K locations or time frames
        rows indexed by group (0,1)
        columns labelled 'pass' and 'fail'
    pass_col : Boolean or string, optional
        if true, column names assumed to be 'pass' and 'fail'
        if string, enter column name of passing counts
        if false, column index 0 interpreted as passing
        column name of failing counts interpreted automatically

    Returns
    ----------
    cmh : cmh chi-squared statistic
    pcmh : p-value of the cmh chi-squared statistic with 1 degree of freedom

    References
    -------

    McDonald, J.H. 2014. Handbook of Biological Statistics (3rd ed.). Sparky
    House Publishing, Baltimore, Maryland.

    Forumla example from Boston University School of Public Health
    https://tinyurl.com/k3du64x
    """
    partial_ed = partial(extract_data, pass_col=pass_col)
    data = list(map(partial_ed, dfs))
    c_nums = [val[2] for val in data]
    c_dens = [val[3] for val in data]
    c_num = sum(c_nums)**2
    c_den = sum(c_dens)
    cmh = float(c_num)/float(c_den)

    pcmh = 1 - chi2.cdf(cmh, 1)
    return cmh, pcmh


def multi_odds_ratio(dfs, pass_col=False):
    """
    Common odds ratio. Designed for multiple interval odds ratio for use with
    the cmh test and breslow-day test, but is generalized for the 2x2 case.

    Parameters
    ------------
    dfs : pd.DataFrame, shape = (2,2) OR list of pd.DataFrame objects
        a K-deep list of 2x2 contingency tables
        representing K locations or time frames
        rows indexed by group (0,1)
        columns labelled 'pass' and 'fail'
    pass_col : Boolean or string, optional
        if true, column names assumed to be 'pass' and 'fail'
        if string, enter column name of passing counts
        if false, column index 0 interpreted as passing
        column name of failing counts interpreted automatically

    Returns
    ----------
    r : pooled odds ratio

    Example
    -------
    from https://en.wikipedia.org/wiki/Odds_ratio#Example

    Of a sample of 100 men and 100 women, 90 men drank wine over the past week,
    while only 10 women did the same.

    df = pd.DataFrame({'pass':[90,20], 'fail':[10,80]})
    multi_odds_ratio(df)
    > 36.0

    For odds ratios over multiple intervals, the use-case of the CMH test,
    let's presume that the next week 70 of 100 men drank wine, but now
    70 of 100 women also drank.

    df2 = pd.DataFrame({'pass':[70,70], 'fail':[30,30]})
    dfs = [df,df2]
    multi_odds_ratio(df)
    > 36.0

    multi_odds_ratio(df2)
    > 1.0

    multi_odds_ratio(dfs)
    >4.043478260869565

    References
    ----------
    Boston University School of Public Health
    https://tinyurl.com/k3du64x
    """

    if isinstance(dfs, list):
        # if we have a list of multiple dfs
        partial_ed = partial(extract_data, pass_col=pass_col)
        data = list(map(partial_ed, dfs))
        r_nums = [val[0] for val in data]
        r_dens = [val[1] for val in data]
        r_num = sum(r_nums)
        r_den = sum(r_dens)
    elif np.shape(dfs) == (2, 2):
        data = extract_data(dfs, pass_col=pass_col)
        r_num = data[0]
        r_den = data[1]
    else:
        return('Input error. Requires 2x2 dataframe or list of dataframes')
    r = float(r_num)/float(r_den)
    return r


def bres_day(df, r, pass_col=False):
    """
    Calculates the Breslow-Day test of homogeneous association for a
    2 x 2 x k table. E.g., given three factors, A, B, and C, the Breslow-Day
    test would measure wheher pairwise effects (AB, AC, BC) have identical
    odds ratios.

    Parameters
    ------------
    df  : pd.DataFrame, shape = (2,2)
        a 2x2 contingency table
        rows indexed by group (0,1)
        columns labelled 'pass' and 'fail'
    r   : odds ratio; auditai.stats.multi_odds_ratio
    pass_col : Boolean or string, optional
        if true, column names assumed to be 'pass' and 'fail'
        if string, enter column name of passing counts
        if false, column index 0 interpreted as passing
        column name of failing counts interpreted automatically

    Returns
    ----------
    bd : Breslow-Day chi-squared statistic
    pcmh : p-value of the cmh chi-squared statistic with 1 degree of freedom

    References
    -------

    """

    pass0, fail0, pass1, fail1, total = parse_matrix(df, pass_col)

    coef = []
    coef.append(1.0-r)
    # coef.append(r*((a+c)+(a+b)) + (d-a))
    # coef.append(r*(-1*(a+c)*(a+b)))
    coef.append(r*((pass0+pass1)+(pass0+fail0)) + (fail1-pass0))
    coef.append(r*(-1*(pass0+pass1)*(pass0+fail0)))

    sols = np.roots(coef)
    if min(sols) > 0:
        t_a = min(sols)
    else:
        t_a = max(sols)

    t_b = (pass0+fail0) - t_a
    t_c = (pass0+pass1) - t_a
    t_d = (fail0+fail1) - t_b

    var = 1/((1/t_a) + (1/t_b) + (1/t_c) + (1/t_d))
    bd = (pass0-t_a)**2/var
    pbd = 1 - chi2.cdf(bd, len(df)-1)

    return bd, pbd


def test_cmh_bd(dfs, pass_col=False):
    """
    Master function for Cochran-Mantel-Haenszel and associated tests
    Overview: Compare multiple 2x2 pass-fail contingency tables by gender
        or ethnicity (pairwise) to determine if there is a consistent
        pattern across regions, time periods, or similar groupings.

    Parameters
    ----------
    dfs : pd.DataFrame, 2x2xK stack of contingency tables
        Cell values are counts of individuals.
        Columns are 'pass' and 'fail'.
        Rows are integer 1 for focal, 0 for reference group.
        K regions, time periods, etc.

    Returns
    -------
    r : common odds ratio
    cmh : Cochran-Mantel-Haenszel statistic (pattern of impact)
    pcmh : p-value of Cochran-Mantel-Haenszel test
    bd : Breslow-Day statistic (sufficiency of common odds ratio)
    pbd : p-value of Breslow-Day test

    Example
    -------
    Handbook of Biological Statistics by John H. McDonald
    Signicant CMH test with non-significant Breslow-Day test
    http://www.biostathandbook.com/cmh.html

    "McDonald and Siebenaller (1989) surveyed allele frequencies at the Lap
    locus in the mussel Mytilus trossulus on the Oregon coast. At four
    estuaries, we collected mussels from inside the estuary and from a marine
    habitat outside the estuary. There were three common alleles and a couple
    of rare alleles; based on previous results, the biologically interesting
    question was whether the Lap94 allele was less common inside estuaries,
    so we pooled all the other alleles into a "non-94" class."

    "There are three nominal variables: allele (94 or non-94), habitat
    (marine or estuarine), and area (Tillamook, Yaquina, Alsea, or Umpqua).
    The null hypothesis is that at each area, there is no difference in the
    proportion of Lap94 alleles between the marine and estuarine habitats."

    tillamook = pd.DataFrame({'Marine':[56,40],'Estuarine':[69,77]})
    yaquina = pd.DataFrame({'Marine':[61,57],'Estuarine':[257,301]})
    alsea = pd.DataFrame({'Marine':[73,71],'Estuarine':[65,79]})
    umpqua = pd.DataFrame({'Marine':[71,55],'Estuarine':[48,48]})
    dfs = [tillamook,yaquina,alsea,umpqua]

    test_cmh_bd(dfs)
    > (1.3174848702393571,
      5.320927767938446,
      0.021070789938349432,
      0.5294859090315444,
      0.9123673420971026)

    """

    r = multi_odds_ratio(dfs, pass_col)
    cmh, pcmh = cmh_test(dfs, pass_col)
    part_bd = partial(bres_day, r=r, pass_col=pass_col)

    # sum of Breslow-Day chi-square statistics
    bd = DataFrame(list(map(part_bd, dfs)))[0].sum()
    pbd = 1 - chi2.cdf(bd, len(dfs)-1)

    return r, cmh, pcmh, bd, pbd
