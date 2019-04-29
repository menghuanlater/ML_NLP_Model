#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     naive_bayes_model
   Description :
   Author :       anlin
   date：          2019-04-29
-------------------------------------------------
   Change Activity:
                   2019-04-29:
-------------------------------------------------
"""
# import region
import numpy as np


# code region
class NaiveBayes:
    def __init__(self):
        """
        n_classes: the number of y kinds
        features_num: how many feature use for classification
        class_prior_probability: just as its name implies, memorize each class probability
        conditional_probability_list: just as its name implies, Under each of the identified categories, calculate
            the probability that each possible value of each feature, 3-D list
        class_map: a map that cast y_labels to [0, 1, 2, 3...]
        features_map: a map that cast each feature's value to [0, 1, 2, 3...]
        lamda: a const for Laplace smoothing
        """
        self.__n_classes = 0
        self.__features_num = 0
        self.__class_prior_probability_list = None
        self.__conditional_probability_list = None
        # class map
        self.__class_map = {}
        self.__features_map = []
        self.__lamda = 1.0

    def fit(self, train_data_x, train_data_y):
        """
        :param train_data_x: np.data
        :param train_data_y: labels
        :return:
        """
        if type(train_data_x) != np.ndarray:
            train_data_x = np.array(train_data_x)
        if type(train_data_y) != list:
            train_data_y = list(train_data_y)
        merge = np.array([[list(train_data_x[i]), train_data_y[i]] for i in range(0, len(train_data_y))])
        labels_set = set(train_data_y)
        self.__n_classes = len(labels_set)
        self.__class_prior_probability_list = []
        for i, t in enumerate(labels_set):
            self.__class_map[str(t)] = i
            self.__class_prior_probability_list.append((train_data_y.count(t) + self.__lamda) / (len(train_data_y) + self.__n_classes * self.__lamda))

        self.__features_num = len(train_data_x[0])
        self.__conditional_probability_list = [[[] for j in range(self.__features_num)] for i in range(self.__n_classes)]
        for i in range(self.__features_num):
            value_set = set(train_data_x[:, i])
            self.__features_map.append({})
            for j, t in enumerate(value_set):
                self.__features_map[i][str(t)] = j
                for k in labels_set:
                    tmp = (merge[merge[:, 1] == k])[:, 0]
                    values = [m[i] for m in tmp]
                    self.__conditional_probability_list[self.__class_map[k]][i].append(
                        (values.count(t) + self.__lamda) / (len(tmp) + self.__lamda * len(value_set))
                    )

    def predict(self, test_data):
        predict_probability_list = []
        for i in range(self.__n_classes):
            p = self.__class_prior_probability_list[i]
            for j, k in enumerate(test_data):
                p *= self.__conditional_probability_list[i][j][self.__features_map[j][str(k)]]
            predict_probability_list.append(p)
        y = predict_probability_list.index(max(predict_probability_list))
        for i in self.__class_map.keys():
            if self.__class_map[i] == y:
                return i


if __name__ == "__main__":
    x = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"], [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"], [3, "L"], [3, "M"],
         [3, "M"], [3, "L"], [3, "L"]]
    y = ["坏", "坏", "好", "好", "坏", "坏", "坏", "好", "好", "好", "好", "好", "好", "好", "坏"]
    model = NaiveBayes()
    model.fit(x, y)
    print(model.predict([2, "S"]))
