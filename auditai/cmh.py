from scipy import stats
from functools import partial
import numpy as np
import pandas as pd

def extract_data(df):
    # utility function for cmh test and odds ratio
    # df needs to be a 2x2 data frame with columns 'pass' and 'fail'
    a = float(df.iloc[0]['pass'])
    d = float(df.iloc[1]['fail'])
    b = float(df.iloc[0]['fail'])
    c = float(df.iloc[1]['pass'])
    t = a + b + c + d

    r_num = (a*d)/(t)
    r_den = (b*c)/(t)
    
    c_num = a - ((a + b)*(a + c))/(t)
    c_dem = ((a + b)*(c + d)*(a + c)*(b + d))/(t*t*(t - 1))
    
    return r_num, r_den, c_num, c_dem
    
    
def cmh_test(dfs):
    # cochran-mantel-haenszel test (pairwise special case)
    # dfs is a 2x2xK data frame,
    #     or a K-deep stack of 2x2 contingency tables
    #     representing K locations or time frames
    # each 2x2 contingency table should have 'pass'/'fail' columns
    #     and group membership denoted 1/0 for rows
    data = map(extract_data, dfs)
    c_nums = [val[2] for val in data]
    c_dems = [val[3] for val in data]
    c_num = sum(c_nums)**2
    c_dem = sum(c_dems)
    cmh = float(c_num)/float(c_dem)
    
    pcmh = 1 - stats.chi2.cdf(cmh, 1)
    return cmh, pcmh
    
    
def odds_ratio(dfs):
    # common odds ratio for use with the cmh test and breslow-day test
    # dfs is a 2x2xK data frame,
    #     or a K-deep stack of 2x2 contingency tables
    #     representing K locations or time frames
    # each 2x2 contingency table should have 'pass'/'fail' columns
    #     and group membership denoted 1/0 for rows
    data = map(extract_data, dfs)
    r_nums = [val[0] for val in data]
    r_dems = [val[1] for val in data]
    r_num = sum(r_nums)
    r_dem = sum(r_dems)
    r = float(r_num)/float(r_dem)
    return r
    
    
def bres_day(r, df):
    # breslow-day test
    # r is a common odds ratio to test (e.g. output of odds_ratio)
    # df is a 2x2 contingency table 
    # df should have 'pass'/'fail' columns
    #     and group membership denoted 1/0 for rows

    a = float(df.iloc[0]['pass'])
    d = float(df.iloc[1]['fail'])
    b = float(df.iloc[0]['fail'])
    c = float(df.iloc[1]['pass'])
    
    coef = [] 
    coef.append(1.0-r)
    coef.append(r*((a+c)+(a+b)) + (d-a))
    coef.append(r*(-1*(a+c)*(a+b)))
    
    sols = np.roots(coef)
    if min(sols) > 0:
        t_a = min(sols)
    else:
        t_a = max(sols)
        
    t_b = (a+b) - t_a
    t_c = (a+c)-t_a
    t_d = (b+d) - t_b
     
    var = 1/((1/t_a) + (1/t_b) + (1/t_c) + (1/t_d))
    bd = (a-t_a)**2/var
    
    return bd


def test_cmh_bd(dfs):
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
    """

    # todo: proper comment here
    # call cmh, odds ratio and breslow day functions
    
    r = odds_ratio(dfs)
    cmh, pcmh = cmh_test(dfs)
    part_bd = partial(bres_day, r)
    bd = sum(map(part_bd, dfs))
    pbd = 1 - stats.chi2.cdf(bd, len(dfs)-1)

    return r, cmh, pcmh, bd, pbd
    