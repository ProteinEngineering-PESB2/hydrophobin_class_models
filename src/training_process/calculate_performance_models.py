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
    Estimate performance for classification and regression models
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

class Performance(object):

    def get_class_performances(
            self, 
            y_true=None, 
            y_predict=None):

        accuracy_value = accuracy_score(y_true, y_predict)
        f1_score_value = f1_score(y_true, y_predict, average='weighted')
        precision_values = precision_score(y_true, y_predict, average='weighted')
        recall_values = recall_score(y_true, y_predict, average='weighted')

        row = [accuracy_value, f1_score_value, precision_values, recall_values]
        return row
    
    #function to process average performance in cross val training process
    def process_performance_cross_val(
            self, 
            performances=None, 
            keys=None):
        
        row_response = []
        for i in range(len(keys)):
            value = np.mean(performances[keys[i]])
            row_response.append(value)
        return row_response

    def get_regrex_performances(
            self,
            y_true=None,
            y_predict=None):
        
        r2_score_value = r2_score(y_true, y_predict)
        mean_absolute_error_value = mean_absolute_error(y_true, y_predict)
        mean_squared_error_value = mean_squared_error(y_true, y_predict, squared=False)

        row = [r2_score_value, mean_absolute_error_value, mean_squared_error_value]
        
        return row
