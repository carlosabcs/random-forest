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
    dt = DecisionTree(
        attributes = [col for col in df.columns if col != 'Joga'],
        target_attribute = 'Joga'
    )
    dt.train(df)


def main():
    random.seed(1)
    test_benchmark()


if __name__ == "__main__":
    main()

