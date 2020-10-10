import numpy as np
import random


class Bootstrap:
    def __init__(self, n_bootstrap, percentage = 63.2):
        self.percentage = percentage
        self.n_bootstrap = n_bootstrap


    def __is_valid_bootstrap(self, size_df, unique_index_train):
        if ((len(unique_index_train) * 100) / size_df) >= self.percentage:
            return True
        return False


    def __generate_random_index(self, size_df, percentage):
        index_original_df = list(range(0, size_df))
        unique_index_train = []

        while not self.__is_valid_bootstrap(size_df,unique_index_train):
            index_train = random.choices(index_original_df, k = size_df)
            unique_index_train = np.unique(index_train)

        index_test = np.delete(index_original_df,unique_index_train)
        return index_train,index_test


    def generate_bootstraps(self, df):
        size_df = len(df)
        dic_boot = {}
        array_index_test = []
        for i in range(0, self.n_bootstrap):
            index_bootstrap =  self.__generate_random_index(size_df,63.2)
            dic_boot[i]= index_bootstrap[0]
            array_index_test = np.concatenate((array_index_test,index_bootstrap[1]))
        return dic_boot,list(dict.fromkeys(array_index_test))
