import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from decision_tree import DecisionTreeClassifier


def test_benchmark():
    df = pd.read_csv('./dados_benchmark.csv', sep=';')
    df['Joga'] = df['Joga'].apply(
        lambda x: 1 if x == 'Sim' else 0
    )
    dt = DecisionTreeClassifier(
        attributes = [col for col in df.columns if col != 'Joga'],
        target_attribute = 'Joga'
    )
    dt.train(df)
    print(
        dt.predict({'Tempo':'Nublado', 'Temperatura':'Quente', 'Umidade':'Alta', 'Ventoso':'FALSO'}),
        ' == ',
        1
    )
    print(
        dt.predict({'Tempo':'Chuvoso', 'Temperatura':'Quente', 'Umidade':'Alta', 'Ventoso':'Verdadeiro'}),
        ' == ',
        0
    )
    print(
        dt.predict({'Tempo':'Ensolarado', 'Temperatura':'Quente', 'Umidade':'Normal', 'Ventoso':'Falso'}),
        ' == ',
        1
    )


def main():
    random.seed(1)
    test_benchmark()


if __name__ == "__main__":
    main()

