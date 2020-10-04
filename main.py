import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from decision_tree import DecisionTreeClassifier


def test_benchmark():
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

