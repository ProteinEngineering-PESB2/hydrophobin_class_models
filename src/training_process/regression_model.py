#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 04 01 2023
# @authors: David Medina Ortiz
# @contact: <david.medina@umag.cl>
# Interpretable and explainable predictive machine learning models for data-driven protein engineering
# Released under MIT License

# Interpretable and explainable predictive machine learning models for data-driven protein engineering
# David Medina-Ortiz1,2, Ashkan Khalifeh3, Hoda Anvari-Kazemabad3, Mehdi D. Davari3,*
# 1 Departamento de Ingenieria En Computacion, Universidad de Magallanes, Avenida Bulnes 01855, Punta Arenas, Chile.
# 2 Centre for Biotechnology and Bioengineering, CeBiB, Beauchef 851, Santiago, Chile
# 3 Department of Bioorganic Chemistry, Leibniz Institute of Plant Biochemistry, Weinberg 3, 06120 Halle, Germany
# *Corresponding author

"""
Supervised learning algorithms to train regression models
"""

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression, TweedieRegressor, PoissonRegressor, GammaRegressor
from sklearn.linear_model import SGDRegressor, RANSACRegressor, HuberRegressor, TheilSenRegressor, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor

from sklearn.model_selection import cross_validate
from calculate_performance_models import Performance
from joblib import dump

class RegressionModel(object):

    def __init__(
            self, 
            X_train=None, 
            y_train=None,
            X_test=None,
            y_test=None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.scores = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'r2']
        self.keys = ['fit_time', 'score_time', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_neg_median_absolute_error', 'test_neg_root_mean_squared_error', 'test_r2']

        self.rgx_model = None
    
    def apply_kernel_ridge(self):
        self.rgx_model = KernelRidge()
    
    def apply_gaussian_process(self):
        self.rgx_model = GaussianProcessRegressor()
    
    def apply_bayesian_regression(self):
        self.rgx_model = BayesianRidge()
    
    def apply_ardr_regression(self):
        self.rgx_model = ARDRegression()
    
    def apply_tweedie_regression(self):
        self.rgx_model = TweedieRegressor()
    
    def apply_poisson_regression(self):
        self.rgx_model = PoissonRegressor()
    
    def apply_gamma_Regression(self):
        self.rgx_model = GammaRegressor()
    
    def apply_sgd_regression(self):
        self.rgx_model = SGDRegressor()
    
    def apply_ransac_regression(self):
        self.rgx_model = RANSACRegressor()
    
    def apply_huber_regression(self):
        self.rgx_model = HuberRegressor()
    
    def apply_theilsen_regression(self):
        self.rgx_model = TheilSenRegressor()
    
    def apply_quantile_regression(self):
        self.rgx_model = QuantileRegressor()

    def apply_gradient(self):
        self.rgx_model = GradientBoostingRegressor()
    
    def apply_hist_gradient(self):
        self.rgx_model = HistGradientBoostingRegressor()
    
    def apply_adaboost(self):
        self.rgx_model = AdaBoostRegressor()
    
    def apply_random_forest(self):
        self.rgx_model = RandomForestRegressor()
    
    def apply_extra_trees(self):
        self.rgx_model = ExtraTreesRegressor()
    
    def apply_bagging(self):
        self.rgx_model = BaggingRegressor()

    def apply_decision_tree(self):
        self.rgx_model = DecisionTreeRegressor()
    
    def apply_svr(self):
        self.rgx_model = SVR()
    
    def apply_knn(self):
        self.rgx_model = KNeighborsRegressor()
    
    def apply_pls(self):
        self.rgx_model = PLSRegression()
    
    def apply_XGBoost(self):
        self.rgx_model = XGBRegressor()

    def training_model(self, cv=None):

        performance_instance = Performance()

        self.rgx_model.fit(self.X_train, self.y_train)

        row_response = [self.rgx_model.__class__.__name__]        

        if cv != None:
            response_cv = cross_validate(
                self.rgx_model, 
                self.X_train, 
                self.y_train, 
                cv=cv, 
                scoring=self.scores)

            cv_performances = performance_instance.process_performance_cross_val(
                performances=response_cv,
                keys=self.keys
            )
            row_response = row_response + cv_performances

        validation_performances = performance_instance.get_regrex_performances(
            y_predict=self.rgx_model.predict(self.X_test),
            y_true=self.y_test
        )

        row_response = row_response + validation_performances

        return row_response

    def export_model_instance(self, name_export="rgx_model.joblib"):

        dump(self.rgx_model, name_export)