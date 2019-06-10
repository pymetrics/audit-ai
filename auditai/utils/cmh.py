#  helper functions specifically for cmh tests


def parse_matrix(df, pass_col=False):
    """
    Utility function for parsing matrix into cell counts. Used for
    parsing strata for the Cochran-Manzel-Haenszel and Breslow-Day tests in
    stats.cmh_test and stats.multi_odds_ratio

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

    # if True or string
    if pass_col:
        # pass a string for pass column
        if isinstance(pass_col, str):
            fail_col = df.columns[df.columns != pass_col][0]
        else:
            # default mapping of column headers
            pass_col = 'pass'
            fail_col = 'fail'
    else:
        pass_col = 1
        fail_col = 0
    pass0 = float(df.iloc[0][pass_col])
    fail0 = float(df.iloc[0][fail_col])
    pass1 = float(df.iloc[1][pass_col])
    fail1 = float(df.iloc[1][fail_col])

    total = pass0 + fail0 + pass1 + fail1

    return pass0, fail0, pass1, fail1, total


def extract_data(df, pass_col=False):
    """
    Utility function for preprocessing data for stats.cmh_test and
    stats.multi_odds_ratio. Used for parsing strata for the
    Cochran-Manzel-Haenszel and Breslow-Day tests in
    stats.cmh_test and stats.multi_odds_ratio Returns component
    values for relevant tests.

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
    r_num : odds ratio numerator; n group 0 pass * n group 1 fail over total
    r_den : odds ratio denominator; n group 0 fail * n group 1 pass over total
    c_num : CMH numerator
    c_den : CMH denominator
    """

    pass0, fail0, pass1, fail1, total = parse_matrix(df, pass_col)

    r_num = (pass0*fail1)/(total)
    r_den = (fail0*pass1)/(total)

    c_num = pass0 - ((pass0 + fail0)*(pass0 + pass1))/(total)
    c_den = (((pass0 + fail0)*(pass1 + fail1)*(pass0 + pass1)*(fail0 + fail1))
             / (total*total*(total - 1)))

    return r_num, r_den, c_num, c_den
