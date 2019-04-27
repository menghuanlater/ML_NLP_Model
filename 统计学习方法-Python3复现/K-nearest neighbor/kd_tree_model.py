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
import math


# code region
class KdTree:
    def __init__(self, train_data):
        """
        data: data for constructing the kd_tree
        """
        self.__data = train_data

    # return the nearest neighbors node
    def get_k_neighbors(self, k_param):
        """
        :param k_param: the k you should point
        :return: [(x1, y1), (x2, y2), ..., (xk, yk)]
        """
        return


class KNN:
    def __init__(self, n_classes, neighbor_num=1, predict_method="majority_vote"):
        """
        n_classes: evident, the number of label categories
        k:  the amount of choosing neighbors for predicting, default value is 1, it means looking for nearest neighbor
        method: the strategy of predicting class
            1. majority_vote: choose the one with the most voting categories, if same, using gaussian function
            2. distance_weighting: use gaussian function
        kd_tree: a data structure to quickly find the k nearest node
        """
        self.__n_classes = n_classes
        self.__k = neighbor_num
        if predict_method not in ["majority_vote", "distance_weighting"]:
            self.__method = "majority_vote"
            print("the predict method you choose is illegal, the method will be set to majority_vote, if want to"
                  "reset the method, please use affiliated function called as reset_predict_method")
        else:
            self.__method = predict_method
        self.__gaussian_a = 1.0  # gaussian curve height
        self.__gaussian_b = 0.0  # gaussian curve mean
        self.__gaussian_c = 1.0  # gaussian curve half-peak width
        self.__kd_tree = None
        self.__fit_flag = False  # if you can use predict or not
        # self.__gamma = 0.001  #

    # to change predict function
    def reset_predict_method(self, new_method):
        if new_method not in ["majority_vote", "distance_weighting"]:
            print("the predict method you choose is illegal, the method will be set to majority_vote, if want to"
                  "reset the method, please use affiliated function called as reset_predict_method")
        else:
            self.__method = new_method

    # calculate Euclidean distance
    @staticmethod
    def get_euclidean_distance(vector1, vector2):
        return (1/len(vector1)) * math.sqrt(np.sum(np.square(vector1-vector2)))

    # calculate gaussian value
    def __get_gaussian_value(self, vector1, vector2):
        return self.__gaussian_a * math.exp(-(math.pow((self.get_euclidean_distance(vector1, vector2) -
                                                        self.__gaussian_b), 2.0)/(2 * math.pow(self.__gaussian_c, 2.0))))

    # according to the k neighbors, use majority vote to decide class
    def __get_class_by_majority_vote(self, k_neighbors, target):
        """
        k_neighbors: [(x, y)]   ---> x is numpy array, y is label
        target: the predict target features vector
        """
        y_t = [0 for i in range(self.__n_classes)]
        y_t_vectors = [[] for i in range(self.__n_classes)]
        for t in k_neighbors:
            y_t[t[1]] += 1
            y_t_vectors[t[1]].append(t)
        max_vote = max(y_t)
        proposal_labels = list(filter(lambda x: True if y_t[x] == max_vote else False, range(self.__n_classes)))
        if len(proposal_labels) == 1:
            return proposal_labels[0]
        else:
            new_tuple_list = []
            for j in proposal_labels:
                new_tuple_list.extend(y_t_vectors[j])
            return self.__get_class_by_distance_weighting(new_tuple_list, target)

    # according to the k neighbors, use distance weighting
    def __get_class_by_distance_weighting(self, k_neighbors, target):
        """
        k_neighbors: [(x, y)]   ---> x is numpy array, y is label
        target: the predict target features vector
        """
        y_t = [0 for i in range(self.__n_classes)]
        for t in k_neighbors:
            y_t[t[1]] += self.__get_gaussian_value(t[0], target)
        return np.array(y_t).argmax()

    # fit model with train_data to build Kd-tree
    def fit(self, train_data):
        self.__kd_tree = KdTree(train_data)
        self.__fit_flag = True

    # do predict
    def predict(self, target):
        if self.__fit_flag is False:
            print("the KNN model lack of train_data, we now just return a None-Type as the prediction")
            return None
        k_neighbors = self.__kd_tree.get_k_neighbors(self.__k)
        if self.__method == "majority_vote":
            return self.__get_class_by_majority_vote(k_neighbors, target)
        elif self.__method == "distance_weighting":
            return self.__get_class_by_distance_weighting(k_neighbors, target)
        else:
            return self.__get_class_by_majority_vote(k_neighbors, target)


if __name__ == "__main__":
    pass
