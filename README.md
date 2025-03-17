# Multi-Layer Perceptron (MLP) Training and Evaluation

## Overview

This repository contains an implementation of a Multi-Layer Perceptron (MLP) model that can be trained on the MNIST and Fashion-MNIST datasets. The training process is configurable via command-line arguments, and the results, including a confusion matrix, are logged using Weights & Biases (W&B).

## Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install numpy tensorflow wandb argparse scikit-learn keras
```

## Training the Model

To train the MLP, run the following command:

```bash
python main.py --wandb_project myproject --wandb_entity myname --dataset fashion_mnist --epochs 10 --batch_size 32 --loss cross_entropy --optimizer adam --learning_rate 0.001 --num_layers 2 --hidden_size 128 --activation ReLU
```

### Command-line Arguments

The script supports the following command-line arguments:

| Argument                 | Description                                                 | Default Value   |
| ------------------------ | ----------------------------------------------------------- | --------------- |
| `-wp`, `--wandb_project` | W&B project name                                            | "myprojectname" |
| `-we`, `--wandb_entity`  | W&B entity name                                             | "myname"        |
| `-d`, `--dataset`        | Dataset to use (`mnist` or `fashion_mnist`)                 | `fashion_mnist` |
| `-e`, `--epochs`         | Number of training epochs                                   | `1`             |
| `-b`, `--batch_size`     | Batch size                                                  | `4`             |
| `-l`, `--loss`           | Loss function (`mean_squared_error`, `cross_entropy`)       | `cross_entropy` |
| `-o`, `--optimizer`      | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`)     | `sgd`           |
| `-lr`, `--learning_rate` | Learning rate                                               | `0.1`           |
| `-m`, `--momentum`       | Momentum for momentum-based optimizers                      | `0.5`           |
| `-beta`, `--beta`        | Beta for RMSProp optimizer                                  | `0.5`           |
| `-beta1`, `--beta1`      | Beta1 for Adam optimizers                                   | `0.5`           |
| `-beta2`, `--beta2`      | Beta2 for Adam optimizers                                   | `0.5`           |
| `-eps`, `--epsilon`      | Epsilon for optimizers                                      | `1e-6`          |
| `-w_d`, `--weight_decay` | Weight decay                                                | `0.0`           |
| `-w_i`, `--weight_init`  | Weight initialization method (`random`, `Xavier`)           | `random`        |
| `-nhl`, `--num_layers`   | Number of hidden layers                                     | `1`             |
| `-sz`, `--hidden_size`   | Size of hidden layers                                       | `4`             |
| `-a`, `--activation`     | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) | `sigmoid`       |

## Model Training Process

1. **Load Dataset:** MNIST or Fashion-MNIST is loaded based on user input.
2. **Initialize Weights & Biases:** Logs training configuration and results.
3. **Initialize MLP:** The model is created with user-specified parameters.
4. **Train Model:** The MLP is trained on the provided dataset.
5. **Evaluate Model:** Test accuracy is computed, and a confusion matrix is logged to W&B.

## Evaluating the Model

After training, the script evaluates the model on the test set and logs the confusion matrix. You can view the results on your W&B dashboard.

## Example Output

```
Training completed!
Test Accuracy: 0.88
```

## Logging with Weights & Biases

All training runs, hyperparameters, and results are logged using W&B. To visualize results, visit:

```
https://wandb.ai/{your-entity}/{your-project}
```
