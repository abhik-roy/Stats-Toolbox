import numpy as np
import pandas as pd


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def entropy(y):
    class_labels = np.unique(y)
    entropy = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        entropy += -p_cls * np.log2(p_cls)
    return entropy


def gini_index(self, y):
    class_labels = np.unique(y)
    gini = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        gini += p_cls**2
    return 1 - gini


def information_gain(parent, l_child, r_child, mode='entropy'):
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)
    if mode == "gini":
        #gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        gain = gini_index(parent) - (weight_l *
                                     gini_index(l_child) + weight_r*gini_index(r_child))
    else:
        #gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        gain = entropy(parent) - (weight_l*entropy(l_child) +
                                  weight_r*entropy(r_child))
    return gain


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=9, max_depth=11):

        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        #dataset = np.concatenate((X_train, Y_train), axis=1)
        #self.root = self.build_tree(dataset)

    # if we reach the stopping condition we must calculate the value that
    # the terminal node returns when we reach it when classifying a new value
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        max_val = max(Y, key=Y.count)
        return max_val

    def get_best_split(self, dataset, num_samples, num_features):
        # save the feature threshold and left and right data
        # for the best split so it can be easily used by the
        # the main program which calls this subroutine
        best_split = {}
        max_info_gain = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left = np.array(
                    [row for row in dataset if row[feature_index] <= threshold])
                dataset_right = np.array(
                    [row for row in dataset if row[feature_index] > threshold])
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -
                                                 1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    # recursive tree building function that saves the tree as a
    # serialized pickle object
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        vals, counts = np.unique(Y, return_counts=True)
        idx = np.argmax(counts)
        if counts[idx]/len(Y) >= 0.95:
            leaf_value = self.calculate_leaf_value(Y)
            return Node(value=leaf_value)
        if num_samples < self.min_samples_split:
            leaf_value = self.calculate_leaf_value(Y)
            return Node(value=leaf_value)
        if curr_depth > self.max_depth:
            leaf_value = self.calculate_leaf_value(Y)
            return Node(value=leaf_value)
        # IF stopping conditions are not met
        else:
            # find the best split
            best_split = self.get_best_split(
                dataset, num_samples, num_features)
            # check if information is actually gained
            if best_split["info_gain"] > 0:
                subtree1 = self.build_tree(
                    best_split["dataset_left"], curr_depth+1)
                subtree2 = self.build_tree(
                    best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            subtree1, subtree2)
        # else we're out of data
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def predict(self, X):
        predictions = []
        for x in X:
            p = (self.make_prediction(x, self.root))
            predictions.append(p)
        return predictions

    def make_prediction(self, x, node):
        if node.value != None:
            return node.value
        feature_val = x[node.feature_index]
        if feature_val <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
