import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import os.path

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForest
from cross_validator import CrossValidator

DATA_PATH = './data/'

def test_benchmark_categorical():
    print('** Categorical benchmark **')
    df = pd.read_csv(DATA_PATH + 'dados_benchmark.csv', sep=';')
    dt = DecisionTreeClassifier(
        target_attribute = 'Joga',
        n_random_attributes=4
    )
    dt.fit(df)
    dt.print_tree()


def test_benchmark_numerical():
    print('\n** Numerical benchmark **')
    df = pd.read_csv(DATA_PATH + 'dados_benchmark_v2.csv', sep=';')
    dt = DecisionTreeClassifier(
        target_attribute = 'Joga',
        n_random_attributes=4
    )
    dt.fit(df)
    dt.print_tree()


def main():
    parser = argparse.ArgumentParser(description='Random Forest parser')
    parser.add_argument('--opt', help='test-benchmark or test-dataset.', required=True)
    parser.add_argument('--dataset', help='The dataset filename.', default='', required=False)
    parser.add_argument('--target_attribute', help='Target attribute to be predicted.', default='', required=False)
    parser.add_argument('--n_trees', help='The number of trees. The default is 5.', default=5, type=int, required=False)
    parser.add_argument('--n_attributes', help='The number of attributes. The default is the squared root of otal attributes.', default=-1, type=int, required=False)
    parser.add_argument('--k_folds', help='The number of folds for cross validation. The default is 5', default=5, type=int, required=False)
    parser.add_argument('--r', help='The number of repetitions for repeated cross validation. The default is 1', default=1, type=int, required=False)
    args = parser.parse_args()

    if args.opt == 'test-benchmark':
        test_benchmark_categorical()
        test_benchmark_numerical()

    if args.opt == 'test-dataset':
        if args.dataset == '' or not os.path.isfile(DATA_PATH + args.dataset):
            print('Dataset not found.')
            return

        try:
            with open(DATA_PATH + args.dataset[:-3] + 'json', 'r') as filetypes:
                types = json.load(filetypes)
        except:
            print('Dataset types not found, automatic types will be used.')
            types = {}

        data = pd.read_csv(
            DATA_PATH + args.dataset,
            delimiter='\t' if args.dataset[-3:] == 'tsv' else ',',
            dtype=types
        )

        if args.target_attribute not in data.columns:
            print("Target attribute doesn't exist on dataset.")
            return

        n_trees = args.n_trees
        n_random_attributes = args.n_attributes
        if n_random_attributes == -1:
            n_random_attributes = int((len(data.columns) - 1) ** 1/2)

        cv = CrossValidator(
            RandomForest(n_trees, args.target_attribute, n_random_attributes)
        )
        cv.cross_validate(data, args.k_folds, args.r)
        print('\nGlobal accuracy: %.3f (%.3f)' % (cv.accuracy, cv.accuracy_std))


if __name__ == "__main__":
    main()

