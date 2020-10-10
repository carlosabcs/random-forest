import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForest
from cross_validator import CrossValidator

def test_benchmark_categorical():
    df = pd.read_csv('./dados_benchmark.csv', sep=';')
    dt = DecisionTreeClassifier(
        target_attribute = 'Joga',
        n_random_attributes=4
    )
    dt.fit(df)
    dt.print_tree()
    print(
        dt.predict({'Tempo': 'Ensolarado', 'Temperatura': 'Quente', 'Umidade': 'Alta', 'Ventoso': 'Falso'}),
        ' == ',
        'Nao'
    )
    print(
        dt.predict({'Tempo': 'Chuvoso', 'Temperatura': 'Quente', 'Umidade': 'Alta', 'Ventoso': 'Falso'}),
        ' == ',
        'Sim'
    )
    print(
        dt.predict({'Tempo': 'Ensolarado', 'Temperatura': 'Quente', 'Umidade': 'Normal', 'Ventoso': 'Falso'}),
        ' == ',
        'Sim'
    )


def test_benchmark_numerical():
    df = pd.read_csv('./dados_benchmark_v2.csv', sep=';')
    dt = DecisionTreeClassifier(
        target_attribute = 'Joga',
        n_random_attributes=4
    )
    dt.fit(df)
    dt.print_tree()
    print(
        dt.predict({'Tempo':12, 'Temperatura':10, 'Umidade':8, 'Ventoso':1}),
        ' == ',
        'Nao'
    )
    print(
        dt.predict({'Tempo':0, 'Temperatura':10, 'Umidade':8, 'Ventoso':1}),
        ' == ',
        'Sim'
    )
    print(
        dt.predict({'Tempo':12, 'Temperatura':10, 'Umidade':4, 'Ventoso':1}),
        ' == ',
        'Sim'
    )


def test_benchmark_label_encoder():
    df = pd.read_csv('./dados_benchmark.csv', sep=';')
    data_train = df.iloc[:10]
    data_test =df.iloc[10:14]
    print(df.columns[0:4].tolist())
    dt = DecisionTreeClassifier(
        target_attribute = 'Joga',
        n_random_attributes = 2
    )
    dt.fit(data_train)
    dt.print_tree()
    for item in data_test.to_dict(orient='records'):
        print(item['Joga'], ' == ', dt.predict(item))


def main():
    # random.seed(1)
    # test_benchmark_numerical()
    # test_benchmark_categorical()
    # test_benchmark_label_encoder()
    data = pd.read_csv(
        'house-votes-84.tsv',
        delimiter='\t',
        dtype={
            'handicapped-infants': str,
            'water-project-cost-sharing': str,
            'adoption-of-the-budget-resolution': str,
            'physician-fee-freeze': str,
            'el-salvador-adi': str,
            'religious-groups-in-schools': str,
            'anti-satellite-test-ban': str,
            'aid-to-nicaraguan-contras': str,
            'mx-missile': str,
            'immigration': str,
            'synfuels-corporation-cutback': str,
            'education-spending': str,
            'superfund-right-to-sue': str,
            'crime': str,
            'duty-free-exports': str,
            'export-administration-act-south-africa': str,
        }
    )
    n_random_attributes = int((len(data.columns) - 1) ** 1/2)
    rf = RandomForest(6, 'target', n_random_attributes)
    cv = CrossValidator('target', rf)
    cv.cross_validate(data, 5, 1)
    # data_train = data.iloc[:300]
    # data_test = data.iloc[300:]
    # data_test.reset_index(drop=True, inplace=True)
    # print('# random attributes: ', n_random_attributes)
    # rf =
    # rf.fit(data_train)
    # rf.predict(data_test)

    # data = pd.read_csv('wine-recognition.tsv', delimiter='\t', dtype={'target': str})\
    #             .sample(frac=1)\
    #             .reset_index(drop=True)
    # data_train = data.iloc[:80]
    # data_test = data.iloc[80:]
    # data_test.reset_index(drop=True, inplace=True)
    # n_random_attributes = int((len(data.columns) - 1) ** 1/2)
    # print('# random attributes: ', n_random_attributes)
    # rf = RandomForest(8, 'target', n_random_attributes)
    # rf.fit(data_train)
    # rf.predict(data_test)

    # for key, val in rf.dic_tree_generate.items():
    #     val.print_tree()



if __name__ == "__main__":
    main()

