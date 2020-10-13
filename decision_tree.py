import pandas as pd
import numpy as np
import math
import random

class DecisionTree:
    def __init__(self):
        self.output_class = None
        self.target_feature = None
        self.information_gain = None
        self.majority_class = None
        self.childs = {}


    def get_class(self, instance):
        '''
        Gets the predicted class of an unlabelled instance
        through the decision tree
        '''
        if self.output_class is not None:
            return self.output_class

        val = instance[self.target_feature]
        if type(val) != str:
            for key, value in self.childs.items():
                comparable_value = float(key[2:])
                if '0_' in key:
                    if val <= comparable_value:
                        return value.get_class(instance)
                    else:
                        return self.childs['1' + key[1:]].get_class(instance)
                else:
                    # Greater than
                    if val > comparable_value:
                        return value.get_class(instance)
                    else:
                        return self.childs['0' + key[1:]].get_class(instance)
        else:
            if val not in self.childs:
                return self.majority_class
            return self.childs[val].get_class(instance)


    def log_tree(self, level=0):
        '''Prints the tree in a readable way'''
        if self.target_feature:
            print('%s===== %s (%.3f) =====' % (' ' * (level * 5), self.target_feature, self.information_gain))
        if self.output_class is not None:
            print('%sPred. => %s' % (' ' * (level * 5), self.output_class))

        level += 1
        for key, value in self.childs.items():
            print('\n%s%s = %s:' % (' ' * (level * 5), self.target_feature, key))
            value.log_tree(level)



class DecisionTreeClassifier:
    def __init__(self, target_attribute, n_random_attributes):
        self.attributes = []
        self.attributes_outcomes = None
        self.target_attribute = target_attribute
        self.decision_tree = None
        self.n_random_attributes = n_random_attributes


    def __save_possible_attributes_outcomes(self, data):
        '''
        Fills a dictionary with the possible outcomes for each attribute
        '''
        self.attributes_outcomes = {}
        for attribute in self.attributes:
            if data[attribute].dtype == 'object':
                self.attributes_outcomes[attribute] = data[attribute].unique()
            else:
                self.attributes_outcomes[attribute + '_discretized'] = [0, 1]


    def __discretize_attributes(self, data):
        '''
        For each numerical type, adds a discretized version of values
        based on the average
        '''
        new_data = data.copy()
        for col_name, dtype in dict(data.dtypes).items():
            if col_name == self.target_attribute or '_discretized' in col_name:
                continue

            if dtype != 'object':
                mean = data[col_name].mean()
                new_data[col_name + '_discretized'] = data.apply(
                    lambda row: 1 if row[col_name] > mean else 0,
                    axis=1
                )
        return new_data


    def __split_data(self, data, attribute):
        '''
        Splits the data based on the possible outcomes of a certain attribute
        '''
        discretized = False
        if data[attribute].dtype != 'object':
            discretized = True
            mean = data[attribute].mean()
            attribute += '_discretized'
            possible_outcomes = [0, 1]
        else:
            possible_outcomes = data[attribute].unique()
        subsets = []
        for outcome in possible_outcomes:
            subsets.append(
                (
                    outcome if not discretized else str(outcome) + '_' + str(mean),
                    data[data[attribute] == outcome]
                )
            )
        return subsets


    def __choose_random_attributes(self, attributes):
        '''
        Randomnly selects a subset from a list of attributes
        '''
        size = len(attributes)
        if size <= self.n_random_attributes:
            return attributes

        random_attributes = []
        while len(random_attributes) < self.n_random_attributes:
            index = random.randint(0, size - 2)
            if attributes[index] not in random_attributes:
                random_attributes.append(attributes[index])
        return random_attributes


    def __get_general_entropy(self, data):
        '''
        Calculates the entropy of the whole dataset passed to the function
        '''
        total_count = len(data.index)
        possible_outcomes = data[self.target_attribute].unique()
        entropy = 0
        for outcome in possible_outcomes:
            subset_outcome_count = len(data[data[self.target_attribute] == outcome])
            entropy -= (subset_outcome_count / total_count) *\
                        math.log(subset_outcome_count / total_count, 2)
        return entropy


    def __get_local_entropy(self, data, attribute):
        '''
        Calculates the entropy of the data just for a certain attribute
        '''
        total_count = len(data.index)
        entropy = 0
        for outcome in self.attributes_outcomes[attribute]:
            subset = data[data[attribute] == outcome]
            local_count = len(subset) # local count

            possible_subset_outcomes = data[self.target_attribute].unique()
            sub_entropy = 0
            for subset_outcome in possible_subset_outcomes:
                subset_outcome_count = len(subset[subset[self.target_attribute] == subset_outcome])
                if subset_outcome_count == 0:
                    sub_entropy -= 0
                    continue
                sub_entropy -= (subset_outcome_count / local_count) *\
                                math.log(subset_outcome_count / local_count, 2)
            entropy += (local_count/total_count) * sub_entropy
        return entropy


    def __get_best_attribute(self, data, attributes):
        '''
        Calculates and returns the attribute with the largest information gain
        '''
        # Get info of all dataset
        attributes = self.__choose_random_attributes(attributes)
        total_entropy = self.__get_general_entropy(data)
        best_gain = -1
        best_attr = attributes[0]
        for attr in attributes:
            # Get info based on certain attribute
            if data[attr].dtype != 'object':
                local_entropy = self.__get_local_entropy(data, attr + '_discretized')
            else:
                local_entropy = self.__get_local_entropy(data, attr)
            gain = total_entropy - local_entropy
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr, best_gain


    def __generate_decision_tree(self, data):
        '''
        Generates the decision tree for the data
        '''
        node = DecisionTree()

        # Set majority class for node
        node.majority_class = data[self.target_attribute].mode().iloc[0]

        # Discretize all non categorical data
        data = self.__discretize_attributes(data)

        # All tuples are of the same class
        distinct_labels = data[self.target_attribute].unique()
        if len(distinct_labels) == 1:
            node.output_class = distinct_labels[0]
            return node

        # No more attributes, get the most common class
        if len(self.attributes) == 0:
            node.output_class = node.majority_class
            return node

        best_attr, score = self.__get_best_attribute(data, self.attributes)
        self.attributes.remove(best_attr)
        node.target_feature = best_attr
        node.information_gain = score

        subsets = self.__split_data(data, best_attr)
        for value, subset in subsets:
            if len(subset) == 0:
                node.output_class = node.majority_class
            else:
                node.childs[value] = self.__generate_decision_tree(subset)
        return node


    def predict(self, data):
        '''
        Returns the predictions for a test set
        '''
        predictions = []
        for _, row in data.iterrows():
            predictions.append(self.predict_single_instance(row))
        return predictions


    def predict_single_instance(self, instance):
        '''
        Returns the prediction for a single instance
        '''
        if self.decision_tree is None:
            print('Decision tree has not been trained yet!!')
        return self.decision_tree.get_class(instance)


    def fit(self, data):
        '''
        Trains the decision tree for the data
        '''
        self.attributes = [
            col for col in data.columns if col != self.target_attribute
        ]
        self.__save_possible_attributes_outcomes(data)
        tree = self.__generate_decision_tree(data)
        self.decision_tree = tree


    def print_tree(self):
        '''
        Prints the tree generated for the training data
        '''
        if self.decision_tree is None:
            print('Decision tree has not been trained yet!!')
        self.decision_tree.log_tree()
