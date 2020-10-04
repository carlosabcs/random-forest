import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from decision_tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def test_benchmark():
    # data =  pd.read_csv('dados_benchmark.csv', delimiter=";")
    # categorical_feature_mask = data.dtypes==object
    # # filter categorical columns using mask and turn it into a list
    # categorical_cols = data.columns[categorical_feature_mask].tolist()
    # # instantiate labelencoder object
    # le = LabelEncoder()
    # # apply le on categorical feature columns
    # data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
    # data[categorical_cols].head(5)

    # data_train = data.iloc[:10]
    # data_test =data.iloc[10:14]
    # print(data)

    # dt = DecisionTreeClassifier(
    #     attributes = [col for col in data.columns if col != 'Joga'],
    #     target_attribute = 'Joga'
    # )
    # dt.train(data_train)
    # dt.print_tree()
    # for i, row in data_test.iterrows():
    #     print(dt.predict(row))
    df = pd.read_csv('./dados_benchmark_v2.csv', sep=';')
    dt = DecisionTreeClassifier(
        attributes = [col for col in df.columns if col != 'Joga'],
        target_attribute = 'Joga'
    )
    dt.train(df)
    dt.print_tree()
    print(
        dt.predict({'Tempo':6, 'Temperatura':10, 'Umidade':8, 'Ventoso':1}),
        ' == ',
        'Sim'
    )
    print(
        dt.predict({'Tempo':0, 'Temperatura':10, 'Umidade':8, 'Ventoso':10}),
        ' == ',
        'Nao'
    )
    print(
        dt.predict({'Tempo':12, 'Temperatura':10, 'Umidade':4, 'Ventoso':1}),
        ' == ',
        'Sim'
    )


def main():
    random.seed(1)
    test_benchmark()


if __name__ == "__main__":
    main()

