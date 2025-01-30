import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from functools import partial
import json

# Import the train_with_entropy function from your script
from o1_toy import train_with_entropy

# Wrapper function to optimize train_with_entropy
def train_with_entropy_optimized(lambda_penalty, lambda_entropy, dataset_path, embed_dim=768, num_epochs=3):
    """
    Run train_with_entropy with given hyperparameters and return the average loss.
    """
    losses = []

    # Logger to collect losses during training
    def custom_logger(avg_loss):
        losses.append(avg_loss)

    train_with_entropy(
        jsonl_path=dataset_path,
        embed_dim=embed_dim,
        num_epochs=num_epochs,
        lambda_penalty=lambda_penalty,
        lambda_entropy=lambda_entropy,
      #  logger=custom_logger,
    )

    # Return the mean of the average losses across epochs
    return np.mean(losses)

# Paths and hyperparameter ranges
dataset_path = "data_sets/train_unlabeled/mushroom.en-train_nolabel.v1.jsonl"  # Update this path to your dataset
embed_dim = 768
num_epochs = 3
lambda_range = [0.001, 0.1]  # Define the range for hyperparameters

# Grid Search Implementation
def grid_search():
    grid_penalty = np.linspace(lambda_range[0], lambda_range[1], 5)
    grid_entropy = np.linspace(lambda_range[0], lambda_range[1], 5)
    best_loss = float("inf")
    best_params = None
    results = []

    for lambda_penalty in grid_penalty:
        for lambda_entropy in grid_entropy:
            avg_loss = train_with_entropy_optimized(
                lambda_penalty, lambda_entropy, dataset_path, embed_dim, num_epochs
            )
            results.append((lambda_penalty, lambda_entropy, avg_loss))
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = (lambda_penalty, lambda_entropy)

    return best_loss, best_params, results

# Bayesian Optimization Implementation
def bayesian_optimization():
    space = {
        "lambda_penalty": hp.uniform("lambda_penalty", lambda_range[0], lambda_range[1]),
        "lambda_entropy": hp.uniform("lambda_entropy", lambda_range[0], lambda_range[1]),
    }

    trials = Trials()

    def objective(params):
        return train_with_entropy_optimized(
            params["lambda_penalty"], params["lambda_entropy"], dataset_path, embed_dim, num_epochs
        )

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=25,  # Number of iterations
        trials=trials
    )

    return trials, best

# Main execution
if __name__ == "__main__":
    print("Running Grid Search...")
    best_loss_grid, best_params_grid, grid_results = grid_search()
    print("\nGrid Search Results:")
    print(f"Best Loss: {best_loss_grid}")
    print(f"Best Parameters: {best_params_grid}")

    print("\nRunning Bayesian Optimization...")
    trials_bayesian, best_params_bayesian = bayesian_optimization()
    best_loss_bayesian = min([trial["result"]["loss"] for trial in trials_bayesian.results])
    print("\nBayesian Optimization Results:")
    print(f"Best Loss: {best_loss_bayesian}")
    print(f"Best Parameters: {best_params_bayesian}")