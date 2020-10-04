import pandas as pd
import numpy as np
import math

class DecisionTree:
    def __init__(self):
        self.output_class = None
        self.feature = None
        self.information_gain = None
        self.subsets = {}


    def get_class(self, instance):
        if self.output_class is not None:
            return self.output_class

        val = instance[self.feature]
        if type(val) != str:
            for key, value in self.subsets.items():
                comparable_value = float(key[2:])
                if '0_' in key:
                    if val <= comparable_value:
                        return value.get_class(instance)
                    else:
                        return self.subsets['1' + key[1:]].get_class(instance)
                # else:
                #     # Greater than
                #     if val > comparable_value:
                #         return value.get_class(instance)
                #     else:
                #         return self.subsets['0' + key[1:]].get_class(instance)
        else:
            return self.subsets[val].get_class(instance)


    def log_tree(self, level=0):
        if self.feature:
            print('%s===== %s (%.3f) =====' % (' ' * (level * 5), self.feature, self.information_gain))
        if self.output_class is not None:
            print('%sPred. => %s' % (' ' * (level * 5), self.output_class))

        level += 1
        for key, value in self.subsets.items():
            print('\n%s%s = %s:' % (' ' * (level * 5), self.feature, key))
            value.log_tree(level)



class DecisionTreeClassifier:
    def __init__(self, attributes, target_attribute):
        self.attributes = attributes
        self.target_attribute = target_attribute
        self.decision_tree = None


    def __discretize_attributes(self, data):
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


    def __get_general_entropy(self, data):
        total_count = len(data.index)
        possible_outcomes = data[self.target_attribute].unique()
        entropy = 0
        for outcome in possible_outcomes:
            subset_outcome_count = len(data[data[self.target_attribute] == outcome])
            entropy -= (subset_outcome_count / total_count) *\
                        math.log(subset_outcome_count / total_count, 2)
        return entropy


    def __get_local_entropy(self, data, attribute):
        total_count = len(data.index)
        possible_outcomes = data[attribute].unique()

        entropy = 0
        for outcome in possible_outcomes:
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
        # Get info of all dataset
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


    def __generate_decision_tree(self, data, attributes):
        node = DecisionTree()

        # Discretize all non categorical data
        data = self.__discretize_attributes(data)

        # All tuples are of the same class
        distinct_labels = data[self.target_attribute].unique()
        if len(distinct_labels) == 1:
            node.output_class = distinct_labels[0]
            return node

        # No more attributes, get the most common class
        if len(attributes) == 0:
            node.output_class = data[self.target_attribute].mode().iloc[0]
            return node

        best_attr, score = self.__get_best_attribute(data, attributes)
        attributes.remove(best_attr)
        node.feature = best_attr
        node.information_gain = score

        subsets = self.__split_data(data, best_attr)
        for value, subset in subsets:
            if len(subset) == 0:
                node.output_class = subset[self.target_attribute].mode().iloc[0]
            else:
                node.subsets[value] = self.__generate_decision_tree(subset, attributes)
        return node


    def predict(self, instance):
        if self.decision_tree is None:
            print('Decision tree has not been trained yet!!')

        return self.decision_tree.get_class(instance)


    def train(self, data):
        tree = self.__generate_decision_tree(data, self.attributes)
        self.decision_tree = tree


    def print_tree(self):
        if self.decision_tree is None:
            print('Decision tree has not been trained yet!!')
        self.decision_tree.log_tree()
