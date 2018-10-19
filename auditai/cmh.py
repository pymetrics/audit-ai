from scipy.stats import chi2
from functools import partial
import numpy as np
import pandas as pd


def parse_matrix(df,pass_col=False):
    """
    Utility function for parsing matrix into cell counts

    Parameters
    ------------
    df : pd.DataFrame, shape (2,2)
        rows indexed by group (0,1)
        columns labelled 'pass' and 'fail'
    pass_col : Boolean or string, optional
        if true, column names assumed to be 'pass' and 'fail'
        if string, enter column name of passing counts
        if false, column index 0 interpreted as passing
        column name of failing counts interpreted automatically

    Returns
    ----------
    pass0 : counts of passing at row index 0
    fail0 : counts of failing at row index 1
    pass1 : counts of passing at row index 1
    fail1 : counts of failing at row index 1
    total : total n for matrix
    """

    if pass_col: #if True or string
        if isinstance(pass_col, str): #pass a string for pass column
            fail_col = df.columns[df.columns != pass_col][0]
        else: #default mapping of column headers
            pass_col = 'pass'
            fail_col = 'fail'
    else: #
        pass_col = 1
        fail_col = 0
    pass0 = float(df.iloc[0][pass_col]) #a
    fail0 = float(df.iloc[0][fail_col]) #b
    pass1 = float(df.iloc[1][pass_col]) #c
    fail1 = float(df.iloc[1][fail_col]) #d

    total = pass0 + fail0 + pass1 + fail1

    return pass0, fail0, pass1, fail1, total

def extract_data(df, pass_col=False):
    """
    Utility function for preprocessing data for cmh test and odds ratio.
    Not for generalized use. Returns component values for relevant tests.

    Parameters
    ------------
    df : pd.DataFrame, shape (2,2)
        rows indexed by group (0,1)
        columns labelled 'pass' and 'fail'
    pass_col : Boolean or string, optional
        if true, column names assumed to be 'pass' and 'fail'
        if string, enter column name of passing counts
        if false, column index 0 interpreted as passing
        column name of failing counts interpreted automatically

    Returns
    ----------
    r_num : odds ratio numerator; n group 0 pass + n group 1 fail over total
    r_den : odds ratio denominator; n group 0 fail + n group 1 pass over total
    c_num : CMH numerator
    c_den : CMH denominator
    """

    pass0, fail0, pass1, fail1, total = parse_matrix(df,pass_col)

    r_num = (pass0*fail1)/(total)
    r_den = (fail0*pass1)/(total)

    c_num = pass0 - ((pass0 + fail0)*(pass0 + pass1))/(total)
    c_den = ((pass0 + fail0)*(pass1 + fail1)*(pass0 + pass1)*(fail0 + fail1))\
    /(total*total*(total - 1))

    return r_num, r_den, c_num, c_den


def cmh_test(dfs, pass_col=False):
    '''
    The pairwise special case of the Cochran-Mantel-Haenszel test. The CMH test
    is a generalization of the McNemar Chi-Squared test of Homogenaity. Whereas
    the McNemar test examines differences over two intervals (usually before
    and after), the CMH test examines differences over any number of
    K instances.

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

    References needed for this calculation of odds ratio (correct but atypical
    formula) and pooled odds ratio.

    Example

    '''
    partial_ed =  partial(extract_data, pass_col= pass_col)
    data = map(partial_ed, dfs)
    c_nums = [val[2] for val in data]
    c_dens = [val[3] for val in data]
    c_num = sum(c_nums)**2
    c_den = sum(c_dens)
    cmh = float(c_num)/float(c_den)

    pcmh = 1 - chi2.cdf(cmh, 1)
    return cmh, pcmh


def odds_ratio(dfs, pass_col=False):
    '''
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

    References
    -------

    References needed for this calculation of odds ratio (correct but atypical
    formula) and pooled odds ratio.

    Example
    -------
    from https://en.wikipedia.org/wiki/Odds_ratio#Example

    Of a sample of 100 men and 100 women, 90 men drank wine over the past week,
    while only 10 women did the same.

    df = pd.DataFrame({'pass':[90,20], 'fail':[10,80]})
    odds_ratio(df)
    > 36.0

    For odds ratios over multiple intervals, the use-case of the CMH test, let's
    presume that the next week 70 of 100 men drank wine, but now 70 of 100 women
    also drank.

    df2 = pd.DataFrame({'pass':[70,70], 'fail':[30,30]})
    dfs = [df,df2]
    odds_ratio(df)
    > 36.0

    odds_ratio(df2)
    > 1.0

    odds_ratio(dfs)
    >4.043478260869565
    '''

    if isinstance(dfs,list):
        #if we have a list of multiple dfs
        partial_ed =  partial(extract_data, pass_col= pass_col)
        data = map(partial_ed, dfs)
        # data = map(extract_data, dfs)
        r_nums = [val[0] for val in data]
        r_dens = [val[1] for val in data]
        r_num = sum(r_nums)
        r_den = sum(r_dens)
    elif np.shape(dfs) == (2,2):
        data = extract_data(dfs, pass_col= pass_col)
        r_num = data[0]
        r_den = data[1]
    else:
        return('Input error. Requires 2x2 dataframe or list of dataframes')
    r = float(r_num)/float(r_den)
    return r


def bres_day(df, r, pass_col=False):
    '''
    Calculates the Breslow-Day test of homogeneous association for a
    2 x 2 x k table. E.g., given three factors, A, B, and C, the Breslow-Day
    test would measure wheher pairwise effects (AB,AC, BC) have identical
    odds ratios.

    Parameters
    ------------
    df  : pd.DataFrame, shape = (2,2)
        a 2x2 contingency table
        rows indexed by group (0,1)
        columns labelled 'pass' and 'fail'
    r   : odds ratio; cmh.odds_ratio
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

    References needed for this calculation of odds ratio (correct but atypical
    formula) and pooled odds ratio.

    Example

    '''

    pass0, fail0, pass1, fail1, total = parse_matrix(df,pass_col)

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
    Cochran-Mantel-Haenszel and associated tests
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

    r = odds_ratio(dfs, pass_col)
    cmh, pcmh = cmh_test(dfs, pass_col)
    part_bd = partial(bres_day, r=r, pass_col=pass_col)
    
    # sum of Breslow-Day chi-square statistics
    bd = pd.DataFrame(map(part_bd, dfs))[0].sum()
    pbd = 1 - chi2.cdf(bd, len(dfs)-1)

    return r, cmh, pcmh, bd, pbd
