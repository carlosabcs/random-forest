import collections
import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier
from bootstrap import Bootstrap


class RandomForest:
    def __init__(self, n_trees, target_name, number_random_attributes):
        self.n_trees = n_trees
        self.dic_tree_generate = {}
        self.target_name = target_name
        self.number_random_attributes = number_random_attributes


    def __max_value(self, list_prediction):
        count = collections.Counter(list_prediction)
        mayor = 0
        mayor_value = None
        for key in count:
            if(mayor < count[key]):
                mayor = count[key]
                mayor_value = key
        return mayor_value


    def __get_index_target(self, df):
        columns_attributes = df.columns.tolist()
        for i in range(0,len(columns_attributes)):
            if columns_attributes[i] == self.target_name:
                return i
        return -1


    def __votation(self, df, index):
        target_index = self.__get_index_target(df)
        list_index_columns = range(0, len(df.columns.tolist()))
        remaining_indexes = np.delete(list_index_columns, target_index)
        #print(remaining_index, dic_index_columns[self.target_name])
        dic_values_predicted = {}

        for key in self.dic_tree_generate:
            y_pred = self.dic_tree_generate[key].predict(df.iloc[index, remaining_indexes])
            target = df.iloc[index, target_index]
            #print(target)
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


    def __comparacion(self, index_test,list_target, list_prediction):
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


    def __create_trees(self, df, n_baggings):
        for i in n_baggings:
            #clf = tree.DecisionTreeClassifier()
            #clf.fit(df.iloc[index_train,:len(df.columns)-1],df.iloc[index_train,-1:])
            clf = DecisionTreeClassifier(self.target_name,self.number_random_attributes)
            clf.fit(df)
            self.dic_tree_generate[i] = clf


    def fit(self, data):
        data.reset_index(drop=True, inplace=True)
        bootstrap = Bootstrap(self.n_trees)
        n_baggings, index_test = bootstrap.generate_bootstraps(data)

        self.__create_trees(data, n_baggings)
        result, index_result = self.__votation(data, index_test)
        target =  data[self.target_name].iloc[index_result.tolist()]
        #print(target,result)
        print('Validation Accuracy: ', self.__comparacion(index_result,target,result)[0])


    def predict(self, data):
        data.reset_index(drop=True, inplace=True)
        index_test = data.index.tolist()
        result, index_result = self.__votation(data, index_test)
        target =  data[self.target_name].iloc[index_result.tolist()]
        #print(target,result)
        print('Test Accuracy: ', self.__comparacion(index_result,target,result)[0])
        return self.__comparacion(index_result,target,result)[0], self.__comparacion(index_result,target,result)[1]