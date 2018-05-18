import numpy as np
import pandas as pd


def _num_samples(x):
    """
    Return number of samples in array_like x.
    """
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array_like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """
    Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


class ClassifierWrapper(object):
    """Simple sklearn wrapper for classifiers"""

    def __init__(self, clf):

        if not hasattr(clf, "decision_function"):
            if not hasattr(clf, "predict_proba"):
                raise ValueError("Classifier object has no decision_function"
                                 "or predict_proba attribute")
        self.clf = clf

    def decision_function(self, X):
        if hasattr(self.clf, 'decision_function'):
            return 1 / (1 + np.exp(-self.clf.decision_function(X)))
        elif hasattr(self.clf, 'predict_proba'):
            # assume positive case is maximum value
            positive_col = self.clf.classes_.argmax()
            return self.clf.predict_proba(X)[:, positive_col]
        else:
            raise ValueError(
                'Classifier has no decision_function or predict_proba')


def check_array(array, all_finite=True, allow_bool=True):
    """
    make it what we want
    """
    # do things...

    return np.array(array)


def boolean_array(array, threshold=None):
    if not isinstance(array, np.ndarray):
        raise TypeError("Expected numpy array, got %s" % type(array))

    if np.unique(array).shape[0] == 2:
        return array.astype('bool')

    if threshold:
        return array >= threshold
        print(array)
    else:
        vals = np.unique(array)
        if vals.shape[0] != 2:
            raise ValueError("Expected 2 unique values when "
                             "threshold=None, got %d" % vals.shape[0])
        max_val = np.max(vals)
        return array == max_val


def arrays_check(labels, results, null_vals=None):
    """
    Given two input arrays of same length,
    returns same arrays with missing values (from either) removed from both

    Parameters
    -----------
    labels : array_like
    results : array_like
    null_vals : list-like, optional
        user-specified unwanted values, e.g. ('', 'missing')

    Returns
    ---------
    labels, results : two-tuple of input arrays, without missings

    """
    # require they be the same lengths
    assert len(labels) == len(results), 'input arrays not the same lengths!'

    # create dataframe and use pandas to remove rows containing nulls
    df = pd.DataFrame(list(zip(labels, results)), columns=['label', 'result'])

    # if user passes unwanted values to check for, replace these with NaNs
    if null_vals is not None:
        df.replace(null_vals, np.nan, inplace=True)

    # if no missing values, return original arrays
    if df.isnull().sum().sum() == 0:
        return labels, results

    # otherwise, get rid of rows with missing values and return remaining
    # arrays
    else:
        df.dropna(axis=0, how='any', inplace=True)
        return df.label.values, df.result.values
