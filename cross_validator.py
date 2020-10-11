import pandas as pd
import numpy as np


class CrossValidator():
    def __init__(self, target_attribute, model):
        self.model = model
        self.target_attribute = target_attribute
        self.accuracy = None
        self.accuracy_std = None
        self.f1_score = None
        self.f1_score_std = None


    def get_folds(self, data, k):
        # Get subsets based on the target class
        outcomes = data[self.target_attribute].unique()

        splitted_folds = []
        for i, outcome in enumerate(outcomes):
            subset = data[data[self.target_attribute] == outcome]
            # Split the subsets into stratified sub-subsets
            splitted_folds.append(
                np.split(
                    subset.sample(frac=1),
                    [ int((1 / k) * i * len(subset)) for i in range(1, k) ]
                )
            )
        # Concat stratified subsets to create stratified sets
        folds = []
        for i in range(k):
            stratified_subsets = []
            for k in range(len(outcomes)):
                stratified_subsets.append(splitted_folds[k][i])
            folds.append(
                pd.concat(stratified_subsets)
            )
        return folds


    def cross_validate(
        self,
        data,
        k_folds,
        r = 1
    ):
        print(
            '===== RF with n_trees = %s and n_attributes = %s =====' % (
                self.model.n_trees,
                self.model.number_random_attributes
            )
        )
        global_acc_list = []
        for it in range(r):
            if r > 1:
                print('ITERATION %s:' % (it + 1))
            folds = self.get_folds(data, k_folds)
            acc_list = []
            for i in range(len(folds)):
                # Train data is composed by all folds except the current one
                train_data = pd.concat(folds[:i] + folds[i+1:])
                self.model.fit(train_data)
                # Test data is composed by current fold
                test_data = folds[i]
                accuracy, _ = self.model.predict(test_data)
                acc_list.append(accuracy)
                global_acc_list.append(accuracy)
                if r == 1:
                    print('Fold %s: acc(%.3f)' % (
                        i + 1, accuracy,
                    ))
            print('Average accuracy: %.3f (%.3f)' % (
                np.mean(acc_list), np.std(acc_list)
            ))
        self.accuracy = np.mean(global_acc_list)
        self.accuracy_std = np.std(global_acc_list)
