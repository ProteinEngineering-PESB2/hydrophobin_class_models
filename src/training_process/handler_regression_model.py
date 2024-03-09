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
Training regression models using supervised learning algorithms. This class contain modules to train
individually regression models and to explore with default hyperparameters
"""
from regression_model import RegressionModel
from handler_models import HandlerModels
import pandas as pd

class HandlerRegressionModels(HandlerModels):

    def __init__(
            self,
            dataset=None, 
            response=None, 
            test_size=None, 
            cv=None):
        
        super().__init__(
            dataset=dataset,
            response=response,
            test_size=test_size,
            cv=cv
        )

    def training_exploring(self):
        
        matrix_exploring = []

        X_train, X_test, y_train, y_test = self.prepare_dataset()

        self.regression_model_instance = RegressionModel(
            X_test=X_test,
            X_train=X_train,
            y_test=y_test,
            y_train=y_train
        )

        try:
            print("Apply apply_gaussian_process")
            self.regression_model_instance.apply_gaussian_process()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_bayesian_regression")
            self.regression_model_instance.apply_bayesian_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_sgd_regression")
            self.regression_model_instance.apply_sgd_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_gradient")
            self.regression_model_instance.apply_gradient()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_hist_gradient")
            self.regression_model_instance.apply_hist_gradient()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_adaboost")
            self.regression_model_instance.apply_adaboost()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_random_forest")
            self.regression_model_instance.apply_random_forest()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_extra_trees")
            self.regression_model_instance.apply_extra_trees()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_bagging")
            self.regression_model_instance.apply_bagging()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_decision_tree")
            self.regression_model_instance.apply_decision_tree()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_svr")
            self.regression_model_instance.apply_svr()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_knn")
            self.regression_model_instance.apply_knn()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_pls")
            self.regression_model_instance.apply_pls()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass
        
        try:
            print("Apply apply_XGBoost")
            self.regression_model_instance.apply_XGBoost()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
            matrix_exploring.append(response_training)
        except:
            pass

        header = []

        if self.cv == None:
            header = ["algorithm", 'r2_score_value', 'mean_absolute_error_value', 'mean_squared_error_value']
        else:
            header = ["algorithm", 'fit_time', 'score_time', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_neg_median_absolute_error', 'test_neg_root_mean_squared_error', 'test_r2', 'r2_score_value', 'mean_absolute_error_value', 'mean_squared_error_value']

        df_summary = pd.DataFrame(matrix_exploring, columns=header)
        return df_summary

    def training_individual(
            self,
            name_algorithm="knn",
            random_state=42,
            name_export=None):
        
        X_train, X_test, y_train, y_test = self.prepare_dataset(random_state=random_state)

        self.regression_model_instance = RegressionModel(
            X_test=X_test,
            X_train=X_train,
            y_test=y_test,
            y_train=y_train
        )
        
        response_training=None
        if name_algorithm == "xgboost":
            self.regression_model_instance.apply_XGBoost()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "kernel_ridge":
            self.regression_model_instance.apply_kernel_ridge()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "gaussia_process":
            self.regression_model_instance.apply_gaussian_process()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "bayesian":
            self.regression_model_instance.apply_bayesian_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "ardr":
            self.regression_model_instance.apply_ardr_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "tweedie":
            self.regression_model_instance.apply_tweedie_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "poisson":
            self.regression_model_instance.apply_poisson_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "gamma":
            self.regression_model_instance.apply_gamma_Regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "sgd":
            self.regression_model_instance.apply_sgd_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "ransac":
            self.regression_model_instance.apply_ransac_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "huber":
            self.regression_model_instance.apply_huber_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "theilsen":
            self.regression_model_instance.apply_theilsen_regression()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "gradient":
            self.regression_model_instance.apply_gradient()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "hist_gradient":
            self.regression_model_instance.apply_hist_gradient()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "adaboost":
            self.regression_model_instance.apply_adaboost()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "random_forest":
            self.regression_model_instance.apply_random_forest()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "extra_tree":
            self.regression_model_instance.apply_extra_trees()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "bagging":
            self.regression_model_instance.apply_bagging()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "decision_tree":
            self.regression_model_instance.apply_decision_tree()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "svr":
            self.regression_model_instance.apply_svr()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        elif name_algorithm == "knn":
            self.regression_model_instance.apply_knn()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        else:
            self.regression_model_instance.apply_pls()
            response_training = self.regression_model_instance.training_model(cv=self.cv)
        
        if name_export != None:
            self.regression_model_instance.export_model_instance(
                name_export=name_export
            )
        
        return response_training