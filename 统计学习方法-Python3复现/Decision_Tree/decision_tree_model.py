#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     decision_tree_model
   Description :  feature selection in [IG, IGR, Gini], pruning use post-pruning,
                  support CART
   Author :       anlin
   date：          2019-05-08
-------------------------------------------------
   Change Activity:
                   2019-05-08:
-------------------------------------------------
"""
# import area


# code area
class DecisionTree:
    def __init__(self, pruning_alpha=0.0, feature_selection="IG", model_use="classification", need_pruning=False):
        """
        :param pruning_alpha: post-pruning procedure --> regularize maximum likelihood estimation parameter
               default value is 0.0, but not recommend, because it's extremely easy to over-fitting
        :param feature_selection: use which calculate formula to choose partition feature
               1. IG: means information gain
               2. IGR: meas information gain rate
               3. Gini: meas Gini index and use CART to generate DT
               default value is IG, and if you input out of range value, likewise the value will be set to IG
        :param model_use: use as classification or regression
               default value is classification. if you set this value as regression, use as CART
        :param need_pruning: True or False
        """
        if pruning_alpha <= 0.0:
            print("the pruning alpha is invalid, we will use default value 0.0")
            self.__alpha = 0.0
        else:
            self.__alpha = pruning_alpha
        if feature_selection not in ["IG", "IGR", "Gini"]:
            print("the feature selection is invalid, we will use default value IG")
            self.__formula = "IG"
        else:
            self.__formula = feature_selection
        if model_use not in ["classification", "regression"]:
            print("the model use is invalid, we will use default value classification")
            self.__pattern = "classification"
        else:
            self.__pattern = "regression"
        self.__pruning_flag = need_pruning

    # use train_data to build decision_tree
    def fit(self, train_data_x, train_data_y):
        pass

    # to predict
    def predict(self, test_data_x):
        pass


if __name__ == "__main__":
    pass



