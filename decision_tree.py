import pandas as pd
import numpy as np
import math

class Tree:
    def __init__(self):
        self.root = None

    def add_root(self, root):
        self.root = root


class DecisionTreeClassifier:
    def __init__(self, attributes, target_attribute):
        self.attributes = attributes
        self.target_attribute = target_attribute
        self.decision_tree = Tree()


    # def __split_data(self, data, attribute):
    #     possible_outcomes = data[self.target_attribute].unique()
    #     subsets = []
    #     for outcome in possible_outcomes:
    #         subsets.append(data[ou])


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


    def __get_best_attribute(self, data):
        # TODO: How to treat continuous values?
        # Get info of all dataset
        total_entropy = self.__get_general_entropy(data)

        best_gain = -1
        best_attr = self.attributes[0]
        for attr in self.attributes:
            # Get info based on certain attribute
            local_entropy = self.__get_local_entropy(data, attr)
            gain = total_entropy - local_entropy
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr


    def train(self, data):
        best_attr = self.__get_best_attribute(data)
        self.decision_tree.add_root({
            'attribute': best_attr,
            'outcomes': []
        })
        # Split into subsets based on attribute
        subsets = self.__split_data(data, best_attr)
        # Hago las particiones y llamo a la función con cada partición
