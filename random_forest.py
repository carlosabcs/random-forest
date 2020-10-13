import collections
import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier
from bootstrap import Bootstrap


class RandomForest:
    def __init__(self, n_trees, target_attribute, n_random_attributes):
        self.n_trees = n_trees
        self.trees = {}
        self.target_attribute = target_attribute
        self.n_random_attributes = n_random_attributes


    def __max_value(self, list_prediction):
        count = collections.Counter(list_prediction)
        max_n = 0
        max_val = None
        for key in count:
            if max_n < count[key]:
                max_n = count[key]
                max_val = key
        return max_val


    def __get_target_index(self, df):
        columns_attributes = df.columns.tolist()
        for i in range(0,len(columns_attributes)):
            if columns_attributes[i] == self.target_attribute:
                return i
        return -1


    def __votation(self, df, index):
        target_index = self.__get_target_index(df)
        list_index_columns = range(0, len(df.columns.tolist()))
        remaining_indexes = np.delete(list_index_columns, target_index)
        dic_values_predicted = {}

        for key in self.trees:
            y_pred = self.trees[key].predict(df.iloc[index, remaining_indexes])
            target = df.iloc[index, target_index]
            df_results = pd.DataFrame()
            df_results['Target'] = target
            df_results['Prediction'] = y_pred
            dic_values_predicted[key] = df_results

        index_prediction = dic_values_predicted[0].index
        dic_values = {}
        for ind in index_prediction:
            list_values = []
            for bagging in dic_values_predicted:
                value_predicted = dic_values_predicted[bagging].loc[ind][1]
                list_values.append(value_predicted)
            dic_values[ind] = list_values

        dic_prediction = {}
        for value in dic_values:
            dic_prediction[value] = self.__max_value(dic_values[value])
        return dic_prediction,index_prediction


    def __calculate_accuracy(self, index_test,list_target, list_prediction):
        aciertos = 0
        count_prediction = 0
        accuracy = 0
        outcomes = []
        for i in index_test:
            if list_target.loc[i] == list_prediction[i]:
                aciertos = aciertos + 1
            if list_prediction[i] != None :
                count_prediction = count_prediction + 1
            accuracy = aciertos / count_prediction * 100
            outcomes.append(list_prediction[i])
        return accuracy ,outcomes


    def __generate_trees(self, df, n_baggings):
        for i in n_baggings:
            clf = DecisionTreeClassifier(
                self.target_attribute,
                self.n_random_attributes
            )
            clf.fit(df)
            self.trees[i] = clf


    def fit(self, data):
        data.reset_index(drop=True, inplace=True)
        bootstrap = Bootstrap(self.n_trees)
        n_baggings, test_index = bootstrap.generate_bootstraps(data)
        self.__generate_trees(data, n_baggings)
        result, result_index = self.__votation(data, test_index)
        target =  data[self.target_attribute].iloc[result_index.tolist()]
        print('Validation Accuracy: ', self.__calculate_accuracy(result_index,target,result)[0])


    def predict(self, data):
        data.reset_index(drop=True, inplace=True)
        test_index = data.index.tolist()
        result, result_index = self.__votation(data, test_index)
        target =  data[self.target_attribute].iloc[result_index.tolist()]
        print('Test Accuracy: ', self.__calculate_accuracy(result_index, target, result)[0])
        return self.__calculate_accuracy(result_index, target, result)[0],\
            self.__calculate_accuracy(result_index, target, result)[1]