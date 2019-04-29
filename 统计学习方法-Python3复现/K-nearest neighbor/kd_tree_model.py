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
class KDNode:
    def __init__(self, divide_axis, point_parent=None, data=None, branch_type=None):
        """
        :param divide_axis: the axis you choose to divide data
        :param point_parent: parent node
        :param branch_type: which branch you belong to your father("left"/"right"/None)
        :param data: memory the divide data(across the divide axis, just one)
        """
        self.__divide_axis = divide_axis
        self.__parent = point_parent
        self.__node_data = data
        self.__left_child = None
        self.__right_child = None
        self.__branch_type = branch_type

    def get_parent_node(self):
        return self.__parent

    def get_divide_axis(self):
        return self.__divide_axis

    def get_left_child(self):
        return self.__left_child

    def get_right_child(self):
        return self.__right_child

    def get_data(self):
        return self.__node_data

    def set_left_child(self, obj):
        self.__left_child = obj

    def set_right_child(self, obj):
        self.__right_child = obj

    def get_branch_type(self):
        return self.__branch_type


class KdTree:
    def __init__(self, train_data_x, train_data_y):
        """
        data: data for constructing the kd_tree, array[(x, y), ...]
        """
        if type(train_data_x) == np.ndarray:
            train_data_x = train_data_x.tolist()
        else:
            train_data_x = list(train_data_x)
        self.__data = [(train_data_x[i], train_data_y[i]) for i in range(len(train_data_y))]
        self.__data.sort(key=lambda x: x[0][0])
        self.__sum_axis = len(train_data_x[0])
        # the root of kd-tree build, the data should be None
        middle_index = int(len(self.__data)/2)
        self.__tree_root = KDNode(divide_axis=0, data=np.array(self.__data[middle_index]))
        self.__tree_root.set_left_child(self.__establish_branch(self.__data[0: middle_index], divide_axis=1 % self.__sum_axis,
                                                                parent_obj=self.__tree_root, branch_type="left"))
        self.__tree_root.set_right_child(self.__establish_branch(self.__data[middle_index+1:], divide_axis=1 % self.__sum_axis,
                                                                 parent_obj=self.__tree_root, branch_type="right"))
        # attributes for global use when finding k neighbor
        # each of calling func get_k_neighbors you should first reset the two variables
        self.__queue = None
        self.__k_number_need_find = 0

    # calculate Euclidean distance
    @staticmethod
    def get_euclidean_distance(vector1, vector2):
        return math.sqrt(np.sum(np.square(vector1 - vector2)))

    # a recurrent function to build the kd-tree
    def __establish_branch(self, data, divide_axis, parent_obj, branch_type):
        if len(data) == 0:
            return None
        middle_index = int(len(data)/2)
        data.sort(key=lambda x: x[0][divide_axis])
        new_node = KDNode(divide_axis=divide_axis, point_parent=parent_obj, data=data[middle_index], branch_type=branch_type)
        new_node.set_left_child(
            self.__establish_branch(data[0: middle_index], divide_axis=(divide_axis+1) % self.__sum_axis,
                                    parent_obj=new_node, branch_type="left"))
        new_node.set_right_child(
            self.__establish_branch(data[middle_index+1:], divide_axis=(divide_axis + 1) % self.__sum_axis,
                                    parent_obj=new_node, branch_type="right"))
        return new_node

    # return the nearest neighbors node
    def get_k_neighbors(self, k_param, target):
        """
        :param k_param: the k you should point
        :param target: aim
        :return: [(x1, y1), (x2, y2), ..., (xk, yk)]  ---> x should change to np.array
        """
        self.__queue = []  # [(x1, y1, dis1), (x2, y2, dis2),..., (x3, y3, dis3)]
        self.__k_number_need_find = k_param

        # first step: find which leaf node space contain target
        p_node = self.__tree_root
        f_node = None
        branch_type = "right"
        while p_node is not None:
            f_node = p_node
            divide_axis = p_node.get_divide_axis()
            if target[divide_axis] <= p_node.get_data()[0][divide_axis]:
                p_node = p_node.get_left_child()
                branch_type = "left"
            else:
                p_node = p_node.get_right_child()
                branch_type = "right"
        self.__queue.append((f_node.get_data()[0], f_node.get_data()[1], self.get_euclidean_distance(f_node.get_data()[0], target)))
        # self.__queue.sort(key=lambda t: t[2])
        if branch_type == "right" and f_node.get_left_child() is not None:
            # when queue have attached the number of k
            if self.__k_number_need_find == len(self.__queue):
                # to check if the determine_circle intersects with child space
                divide_axis = f_node.get_divide_axis()
                axis_middle_value = f_node.get_data()[0][divide_axis]
                if target[divide_axis] - self.__queue[-1][2] <= axis_middle_value:
                    self.__search_neighbor_in_child(f_node.get_left_child(), target)
            else:
                self.__search_neighbor_in_child(f_node.get_left_child(), target)
        elif branch_type == "left" and f_node.get_right_child() is not None:
            if self.__k_number_need_find == len(self.__queue):
                divide_axis = f_node.get_divide_axis()
                axis_middle_value = f_node.get_data()[0][divide_axis]
                if target[divide_axis] - self.__queue[-1][2] >= axis_middle_value:
                    self.__search_neighbor_in_child(f_node.get_right_child(), target)
            else:
                self.__search_neighbor_in_child(f_node.get_right_child(), target)
        if f_node.get_parent_node() is not None:
            self.__search_neighbor_in_parent(f_node.get_parent_node(), f_node.get_branch_type(), target)
        return [(t[0], t[1]) for t in self.__queue]

    # two functions to help func get_k_neighbors
    def __search_neighbor_in_child(self, p_node, target):
        divide_axis = p_node.get_divide_axis()
        axis_middle_value = p_node.get_data()[0][divide_axis]
        # left-child
        if p_node.get_left_child() is not None:
            if self.__k_number_need_find == len(self.__queue):
                if target[divide_axis] - self.__queue[-1][2] <= axis_middle_value:
                    self.__search_neighbor_in_child(p_node.get_left_child(), target)
            else:
                self.__search_neighbor_in_child(p_node.get_left_child(), target)
        # middle-itself
        distance = self.get_euclidean_distance(p_node.get_data()[0], target)
        if self.__k_number_need_find == len(self.__queue) and distance < self.__queue[-1][2]:
            self.__queue[-1] = (p_node.get_data()[0], p_node.get_data()[1], distance)
            self.__queue.sort(key=lambda t: t[2])
        elif self.__k_number_need_find > len(self.__queue):
            self.__queue.append((p_node.get_data()[0], p_node.get_data()[1], distance))
            self.__queue.sort(key=lambda t: t[2])
        # right-child
        if p_node.get_right_child() is not None:
            if self.__k_number_need_find == len(self.__queue):
                if target[divide_axis] - self.__queue[-1][2] >= axis_middle_value:
                    self.__search_neighbor_in_child(p_node.get_right_child(), target)
            else:
                self.__search_neighbor_in_child(p_node.get_right_child(), target)

    def __search_neighbor_in_parent(self, p_node, branch_type, target):
        divide_axis = p_node.get_divide_axis()
        axis_middle_value = p_node.get_data()[0][divide_axis]
        distance = self.get_euclidean_distance(p_node.get_data()[0], target)
        if self.__k_number_need_find == len(self.__queue) and distance < self.__queue[-1][2]:
            self.__queue[-1] = (p_node.get_data()[0], p_node.get_data()[1], distance)
            self.__queue.sort(key=lambda t: t[2])
        elif self.__k_number_need_find > len(self.__queue):
            self.__queue.append((p_node.get_data()[0], p_node.get_data()[1], distance))
            self.__queue.sort(key=lambda t: t[2])
        if branch_type == "left":
            if p_node.get_right_child() is not None:
                if self.__k_number_need_find == len(self.__queue):
                    if target[divide_axis] - self.__queue[-1][2] >= axis_middle_value:
                        self.__search_neighbor_in_child(p_node.get_right_child(), target)
                else:
                    self.__search_neighbor_in_child(p_node.get_right_child(), target)
        elif branch_type == "right":
            if p_node.get_left_child() is not None:
                if self.__k_number_need_find == len(self.__queue):
                    if target[divide_axis] - self.__queue[-1][2] <= axis_middle_value:
                        self.__search_neighbor_in_child(p_node.get_left_child(), target)
                else:
                    self.__search_neighbor_in_child(p_node.get_left_child(), target)
        if p_node.get_parent_node() is not None:
            self.__search_neighbor_in_parent(p_node.get_parent_node(), p_node.get_branch_type(), target)


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
        return math.sqrt(np.sum(np.square(vector1-vector2)))

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
    def fit(self, train_data_x, train_data_y):
        self.__kd_tree = KdTree(train_data_x, train_data_y)
        self.__fit_flag = True

    # do predict
    def predict(self, target):
        if self.__fit_flag is False:
            print("the KNN model lack of train_data, we now just return a None-Type as the prediction")
            return None
        k_neighbors = self.__kd_tree.get_k_neighbors(self.__k, target=target)
        if self.__method == "majority_vote":
            return self.__get_class_by_majority_vote(k_neighbors, target)
        elif self.__method == "distance_weighting":
            return self.__get_class_by_distance_weighting(k_neighbors, target)
        else:
            return self.__get_class_by_majority_vote(k_neighbors, target)


if __name__ == "__main__":
    model = KNN(n_classes=3, neighbor_num=3)
    model.fit(train_data_x=np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]), train_data_y=np.array([1, 2, 1, 2, 1, 1]))
    print(model.predict(target=np.array([11.9, 2.8])))
