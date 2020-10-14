# Random Forest and ensemble learning

*This implementation is part of the course CMP263 Machine Learning, from Instituto de Informática (UFRGS)*

## Installation
This project requires two libraries which can be installed by running: `pip install pandas numpy`

## Execution
The main script receives a set of arguments:
- `--opt`: *test-benchmark* or *test-dataset*. The first one generates the decision tree for benchmark dataset in both versions, categorical and numerical. The second one executes the Random Forest algorithm to a dataset whose name and other parameters need to be passed as arguments too.
- `--dataset`: the filename of the dataset. The dataset needs to be in the same path of the script. This argument is always required when the option is *test-dataset*. It is possible to add also a JSON file with the types of each attribute of the dataset, the file with this information should have the same name than the dataset.
- `--target_attribute`: the name of the attribute to be predicted. This argument is always required when the option is *test-dataset*.
- `--n_trees`: the number of trees to be generated for the random forest algorithm. The default is 5.
- `--n_attributes`: the number of attributes to be used for the decision trees generation. The default is the squared root of total attributes.
- `--k_folds`: the number of folds to be used for cross validation. The default is 5.
- `--r`: the number of repetitions to be used for repeated cross validation. The default is 1.

## Examples

### Testing decision tree generation from benchmark dataset
```bash
python main.py --opt test-benchmark
```

The output should be like:
```bash
** Categorical benchmark **
===== Tempo (0.247) =====

     Tempo = Ensolarado:
     ===== Umidade (0.971) =====

          Umidade = Alta:
          Pred. => Nao

          Umidade = Normal:
          Pred. => Sim

     Tempo = Nublado:
     Pred. => Sim

     Tempo = Chuvoso:
     ===== Ventoso (0.971) =====

          Ventoso = Falso:
          Pred. => Sim

          Ventoso = Verdadeiro:
          Pred. => Nao
```

### Testing datasets training and prediction
For running a repeated cross validation with 2 repetitions and 5 folds, using 8 trees with 6 random attributes, the command to be run should be:
```bash
python main.py --opt test-dataset --dataset house-votes-84.tsv --target_attribute target --n_trees 8 --n_attributes 6 --k_folds 5 --r 2
```

The output must be something like:
```bash
===== RF with n_trees = 8 and n_attributes = 6 =====
ITERATION 1:
- Fold 1:
Validation Accuracy:  98.52507374631269
Test Accuracy:  95.34883720930233
- Fold 2:
Validation Accuracy:  96.16519174041298
Test Accuracy:  95.40229885057471
- Fold 3:
Validation Accuracy:  97.94721407624634
Test Accuracy:  96.55172413793103
- Fold 4:
Validation Accuracy:  97.65395894428153
Test Accuracy:  95.40229885057471
- Fold 5:
Validation Accuracy:  96.71641791044776
Test Accuracy:  94.31818181818183

Average accuracy: 95.405 (0.707)

ITERATION 2:
- Fold 1:
Validation Accuracy:  96.47058823529412
Test Accuracy:  96.51162790697676
- Fold 2:
Validation Accuracy:  97.3293768545994
Test Accuracy:  91.95402298850574
- Fold 3:
Validation Accuracy:  95.26627218934911
Test Accuracy:  95.40229885057471
- Fold 4:
Validation Accuracy:  99.10979228486647
Test Accuracy:  91.95402298850574
- Fold 5:
Validation Accuracy:  93.49112426035504
Test Accuracy:  90.9090909090909

Average accuracy: 93.346 (2.194)

Global accuracy: 94.375 (1.928)
```

