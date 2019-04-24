#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     kd_tree_model
   Description :   base on kd_tree search method model
   Author :       anlin
   date：          2019-04-22
-------------------------------------------------
   Change Activity:
                   2019-04-22:
-------------------------------------------------
"""
# import region
import numpy as np


# code region
class KdTree:
    def __init__(self, train_data):
        """
        data: data for constructing the kd_tree
        """
        self.__data = train_data


class KNN:
    def __init__(self, neighbor_num, predict_method="majority_vote"):
        """
        k:  the amount of choosing neighbors for predicting
        method: the strategy of predicting class
            1. majority vote: choose the one with the most voting categories, if same, using gaussian function
            2. distance_weighting: use gaussian function
        """
        self.__k = neighbor_num
        self.__method = predict_method
        self.__gaussian_a = 1.0  # gaussian curve height
        self.__gaussian_b = 0.0  # gaussian curve mean
        self.__gaussian_c = 1.0  # gaussian curve half-peak width


if __name__ == "__main__":
    pass
