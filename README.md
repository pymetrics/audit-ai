# audit-AI

<img src="https://www.pymetrics.com/static/base/img/branding/favicon.ico">

Open Sourced Bias Testing for Generalized Machine Learning Applications

`audit-AI` is a Python library built on top of `pandas` and `sklearn`that
implements fairness-aware machine learning algorithms. `audit-AI` was developed
by the Data Science team at [pymetrics](https://www.pymetrics.com/)

# Bias Testing for Generalized Machine Learning Applications

`audit-AI` a tool to measure and mitigate the effects discriminatory
patterns in training data and the predictions made by machine learning
algorithms trained for the purposes of socially sensitive decision processes.

The overall goal of this research is to come up with a reasonable way to think
about how to make machine learning algorithms more fair. While identifying
potential bias in training datasets and by consequence the machine learning
algorithms trained on them is not sufficient to solve the problem of
discrimination, in a world where more and more decisions are being automated
by Artifical Intelligence, our ability to understand and identify the degree
to which an algorithm is fair or biased is a step in the right direction.

# Features

Here are a few of the bias testing and algorithm auditing techniques
that this library implements.

### Classification tasks

- 4/5th, fisher, z-test, bayes factor, chi squared
- sim_beta_ratio, classifier_posterior_probabilities

### Regression tasks

- anova
- 4/5th, fisher, z-test, bayes factor, chi squared
- group proportions at different thresholds

# Installation

The source code is currently hosted on GitHub: https://github.com/pymetrics/bias-testing
You can install the latest released version with `pip`.

```
# pip
pip install audit-AI
```

If you install with pip, you'll need to install scikit-learn, numpy, and pandas
with either pip or conda. Version requirements:

- numpy
- scipy
- pandas

For vizualization:
- matplotlib
- seaborn

# How to use this package:

```python

from auditai.misc import bias_test_check

X = df.loc[:,features]
y_pred = clf.predict_proba(X)

# test for bias
bias_test_check(labels=df['gender'], results=y_pred, category='Gender')

>>> *Gender passes 4/5 test, Fisher p-value, Chi-Squared p-value, z-test p-value and Bayes Factor at 50.00*

```
To get a plot of the different tests at different thresholds:

```python

from auditai.viz import plot_threshold_tests

X = df.loc[:,features]
y_pred = clf.predict_proba(X)

# test for bias
plot_threshold_tests(labels=df['gender'], results=y_pred, category='Gender')

```
<img alt="Sample audit-AI Plot" src="data/auditAI_gender_plot.png" width=1600>

# Example Datasets

- [german-credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [student-performance](https://archive.ics.uci.edu/ml/datasets/student+performance)