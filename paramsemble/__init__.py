"""
Paramsemble (Parametric Ensemble Regression) Package

A Python machine learning library for regression tasks that combines automated 
feature selection, baseline comparison, and ensemble modeling.
"""

from paramsemble.estimator import ParamsembleRegressor
from paramsemble._version import __version__

__all__ = ['ParamsembleRegressor', '__version__']
