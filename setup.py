from codecs import open
from os import path
from setuptools import setup, find_packages


# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# load the version
exec(open('auditai/version.py').read())

setup(name='audit-AI',
      version=__version__,
      description='audit-AI detects demographic differences in the output of machine learning models or other assessments',
      long_description=long_description,
      long_description_content_type='text/markdown',
      keywords=[
            "audit",
            "adverse impact",
            "artificial intelligence",
            "machine learning",
            "fairness",
            "bias",
            "accountability",
            "transparency",
            "discrimination",
      ],
      url='https://github.com/pymetrics/audit-ai',
      author='pymetrics Data Team',
      author_email='data@pymetrics.com',
      project_urls={
          'Company': 'https://www.pymetrics.com/science/',
      },
      license='MIT',
      packages=find_packages(exclude=['*.tests', '*.tests.*']),
      install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'statsmodels',
            'sklearn'
      ],
      tests_require=['pytest-cov'],
      extras_require={
            'dev': ['pytest-cov', 'flake8', 'detox', 'mock', 'six']
      },
      classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Financial and Insurance Industry',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Legal Industry',
            'Intended Audience :: Other Audience',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Visualization',
      ])
