import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from decision_tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

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
    random.seed(1)
    test_benchmark_label_encoder()


if __name__ == "__main__":
    main()

