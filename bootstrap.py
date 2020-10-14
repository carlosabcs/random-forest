import numpy as np
import random


class Bootstrap:
    def __init__(self, n_bootstraps, percentage = 63.2):
        self.percentage = percentage
        self.n_bootstraps = n_bootstraps


    def __is_valid_bootstrap(self, size_df, unique_indexes_train):
        '''
        Verify that the number of instances are at least 63.2
        percent of the whole training dataset
        '''
        if ((len(unique_indexes_train) * 100) / size_df) >= self.percentage:
            return True
        return False


    def __generate_random_indexes(self, size_df, percentage):
        '''
        Generate random indexes with repositioning (method = random choise)
        for the training dataset. The remaining indexes are used to the testing
        dataset (validation)
        '''
        indexes_original_df = list(range(0, size_df))
        unique_indexes_train = []

        while not self.__is_valid_bootstrap(size_df, unique_indexes_train):
            indexes_train = random.choices(indexes_original_df, k = size_df)
            unique_indexes_train = np.unique(indexes_train)

        indexes_test = np.delete(indexes_original_df, unique_indexes_train)
        return indexes_train, indexes_test


    def generate_bootstraps(self, df):
        '''
        Generate 'n' bootstraps, for each of them is generated a training and testing
        dataset. However, at the end all test subsets are merged into one (validation)
        '''
        size_df = len(df)
        dic_bootstraps = {}
        final_array_indexes_test = []
        for i in range(0, self.n_bootstraps):
            index_bootstrap =  self.__generate_random_indexes(size_df, 63.2)
            dic_bootstraps[i]= index_bootstrap[0]
            final_array_indexes_test = np.concatenate((final_array_indexes_test, index_bootstrap[1]))
        return dic_bootstraps, list(dict.fromkeys(final_array_indexes_test))
