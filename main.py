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
        dt.predict_single_instance({'Tempo': 'Ensolarado', 'Temperatura': 'Quente', 'Umidade': 'Alta', 'Ventoso': 'Falso'}),
        ' == ',
        'Nao'
    )
    print(
        dt.predict_single_instance({'Tempo': 'Chuvoso', 'Temperatura': 'Quente', 'Umidade': 'Alta', 'Ventoso': 'Falso'}),
        ' == ',
        'Sim'
    )
    print(
        dt.predict_single_instance({'Tempo': 'Ensolarado', 'Temperatura': 'Quente', 'Umidade': 'Normal', 'Ventoso': 'Falso'}),
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
        dt.predict_single_instance({'Tempo': 12, 'Temperatura': 10, 'Umidade': 8, 'Ventoso': 1}),
        ' == ',
        'Nao'
    )
    print(
        dt.predict_single_instance({'Tempo': 0, 'Temperatura': 10, 'Umidade': 8, 'Ventoso': 1}),
        ' == ',
        'Sim'
    )
    print(
        dt.predict_single_instance({'Tempo': 12, 'Temperatura': 10, 'Umidade': 4, 'Ventoso': 1}),
        ' == ',
        'Sim'
    )


def main():
    # random.seed(1)
    test_benchmark_numerical()
    test_benchmark_categorical()
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
    # cv.cross_validate(data, 5, 1)

    data = pd.read_csv(
        'wine-recognition.tsv',
        delimiter='\t'
    )
    n_random_attributes = int((len(data.columns) - 1) ** 1/2)
    rf = RandomForest(8, 'target', n_random_attributes)
    cv = CrossValidator('target', rf)
    cv.cross_validate(data, 3, 1)


if __name__ == "__main__":
    main()

