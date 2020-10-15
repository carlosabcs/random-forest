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
        '''
        Given a list of predictions, choose the prediction that repeats the most
        (majority class)
        '''
        count = collections.Counter(list_prediction)
        max_count = 0
        max_val = None
        for key in count:
            if max_count < count[key]:
                max_count = count[key]
                max_val = key
        return max_val


    def __get_target_index(self, df):
        '''
        Returns the index of the column target
        '''
        columns_attributes = df.columns.tolist()
        for i in range(0,len(columns_attributes)):
            if columns_attributes[i] == self.target_attribute:
                return i
        return -1


    def __majority_votation(self, df, indexes):
        '''
        Given a dataframe and the indexes to be considered for testing, obtain
        the prediction en each tree of the random forest and choose the final
        prediction using the majority votation.
        '''
        target_index = self.__get_target_index(df)
        list_index_columns = range(0, len(df.columns.tolist()))
        remaining_indexes = np.delete(list_index_columns, target_index)
        dic_values_predicted = {}

        for key_bagging in self.trees:
            y_pred = self.trees[key_bagging].predict(df.iloc[indexes, remaining_indexes])
            target_values = df.iloc[indexes, target_index]
            df_results = pd.DataFrame()
            df_results['Target'] = target_values
            df_results['Prediction'] = y_pred
            dic_values_predicted[key_bagging] = df_results

        indexes_prediction = dic_values_predicted[0].index
        dic_values_per_index = {}
        for index in indexes_prediction:
            list_values_predicted = []
            for key_bagging in dic_values_predicted:
                value_predicted = dic_values_predicted[key_bagging].loc[index][1]
                list_values_predicted.append(value_predicted)
            dic_values_per_index[index] = list_values_predicted

        dic_prediction = {}
        for index in dic_values_per_index:
            dic_prediction[index] = self.__max_value(dic_values_per_index[index])
        return dic_prediction, indexes_prediction


    def __calculate_accuracy(self, indexes_test, list_target, list_prediction):
        '''
        Given a list of targets and the prediction generated, calculate the
        final accuracy
        '''
        hits = 0
        count_predictions = 0
        accuracy = 0
        outcomes = []
        for i in indexes_test:
            if list_target.loc[i] == list_prediction[i]:
                hits = hits + 1
            if list_prediction[i] != None :
                count_predictions = count_predictions + 1
            accuracy = hits / count_predictions * 100
            outcomes.append(list_prediction[i])
        return accuracy, outcomes


    def __generate_trees(self, df, n_baggings):
        '''
        Given 'n' baggings, generate one Tree model for each of them
        '''
        for i in n_baggings:
            clf = DecisionTreeClassifier(
                self.target_attribute,
                self.n_random_attributes
            )
            clf.fit(df.iloc[n_baggings[i],:])
            self.trees[i] = clf


    def fit(self, data):
        '''
        Given a dataset, train the Random Forest model
        '''
        data.reset_index(drop=True, inplace=True)
        bootstrap = Bootstrap(self.n_trees)
        n_baggings, test_index = bootstrap.generate_bootstraps(data)
        self.__generate_trees(data, n_baggings)
        result, result_index = self.__majority_votation(data, test_index)
        target =  data[self.target_attribute].iloc[result_index.tolist()]
        print('Validation Accuracy: ', self.__calculate_accuracy(result_index,target,result)[0])


    def predict(self, data):
        '''
        Given a dataset, generate the predictions based on the Random Forest
        model generated previously
        '''
        data.reset_index(drop=True, inplace=True)
        test_index = data.index.tolist()
        result, result_index = self.__majority_votation(data, test_index)
        target =  data[self.target_attribute].iloc[result_index.tolist()]
        print('Test Accuracy: ', self.__calculate_accuracy(result_index, target, result)[0])
        return self.__calculate_accuracy(result_index, target, result)[0],\
            self.__calculate_accuracy(result_index, target, result)[1]