#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     initial_model
   Description :   perceptron model original version
   Author :       anlin
   date：          2019-04-20
-------------------------------------------------
   Change Activity:
                   2019-04-20:
-------------------------------------------------
"""
# import region
import numpy as np


# code region
class Perceptron:
    # class object initial function
    def __init__(self, learning_rate, features_num, max_steps):
        """
        alpha: learning_rate
        dim: feature dimension
        max_step: maximum iteration steps
        """
        self.__alpha = learning_rate
        self.__dim = features_num
        self.__weights = np.random.normal(loc=0.0, scale=1.0, size=features_num)
        self.__bias = 0
        self.__max_step = max_steps

    # according to output value, map the class
    @staticmethod
    def __get_class(value):
        if value > 0:
            return 1
        else:
            return -1

    # update weights and bias
    def __update(self, error_x, real_y):
        self.__weights += np.multiply(self.__alpha, np.multiply(real_y, error_x))
        self.__bias += self.__alpha * real_y

    # fit model with train data(np.data)
    def fit(self, train_data_x, train_data_y):
        train_data_x = np.array(train_data_x)
        train_data_y = np.array(train_data_y, dtype=np.int32)
        for _ in range(self.__max_step):
            next_step_flag = False
            for i, row in enumerate(train_data_x):
                outcome = np.sum(np.multiply(self.__weights, row)) + self.__bias
                obj_class = self.__get_class(outcome)
                if obj_class != train_data_y[i]:
                    self.__update(row, train_data_y[i])
                    next_step_flag = True
                    break
            if next_step_flag:
                continue
            else:
                break

    # predict labels(np.data)
    def predict(self, test_data_x):
        outcome = np.sum(np.multiply(np.array(test_data_x), self.__weights) + self.__bias, axis=1)
        for i, t in enumerate(outcome):
            obj_class = self.__get_class(t)
            outcome[i] = obj_class
        return outcome.astype(np.int32)


if __name__ == "__main__":
    model = Perceptron(learning_rate=1.0, features_num=2, max_steps=1000)
    x_train = [[3, 3], [4, 3], [1, 1], [4, 0], [0, 0], [-1, 8], [1, 10]]
    y_train = [1, 1, -1, 1, -1, -1, 1]
    model.fit(x_train, y_train)
    a = model.predict([[6, 6], [10, 10], [-6, -9]])
    print(a)
