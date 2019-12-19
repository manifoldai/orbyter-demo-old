# -*- coding: utf-8 -*-
"""
This module is for the evaluation of model performance.

Examples:
    Example command line executable::

        $ python evaluate.py config.yml
"""
import logging
import os
import tempfile
from pathlib import Path

import click
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score

from orbyter_demo.util.config import parse_config
from orbyter_demo.util.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument("config_file", type=str, default="/mnt/configs/config.yml")
def evaluate(config_file):
    """
    Main function that trains model and runs evaluation.

    Args:
        config_file (str): path to config file (for logging)

    Returns:
        None
    """
    config = parse_config(config_file)

    # Load data from processed directory
    logger.info(f"Loading training data from {config['global']['processed_data_dir']}.")
    processed_data_dir = Path(config["global"]["processed_data_dir"])
    X = pd.read_parquet(processed_data_dir / "X.pqt")
    y = pd.read_parquet(processed_data_dir / "y.pqt")["MedianHouseValue"]

    # Do cross validation to get average R2 metric
    # hello
    logger.info("5-fold cross validation to get R2 metric.")
    n_estimators = 10
    rf = RandomForestRegressor(n_estimators=n_estimators)
    scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    logger.info(f"Average R2: {scores.mean()}")

    # Do cross validation to get out of sample predictions for training data
    logger.info("5-fold cross validation to generate y vs y_hat plot.")
    y_predicted = cross_val_predict(rf, X, y, cv=5)
    fig = plot_y_yhat(y, y_predicted)

    with tempfile.TemporaryDirectory() as tmpdir:
        scatter_file = Path(tmpdir) / "test.png"
        fig.savefig(scatter_file)

        params = {"model": "random_forest", "n_estimators": n_estimators}
        metrics = {"mean R2": scores.mean()}
        artifacts = {"config_file": config_file, "scatter": scatter_file}
        log_experiment(
            params,
            metrics,
            artifacts,
            config["evaluate"]["experiment_name"],
            os.environ["MLFLOW_TRACKING_URI"],
            os.environ["MLFLOW_ARTIFACT_LOCATION"],
        )


def plot_y_yhat(y_true, y_predicted):
    """
    Plot hexbin plot of ytrue vs ypred for regression.

    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values

    Returns:
        fig (matplotlib.figure)
    """
    sns.set(style="white", color_codes=True)
    sns.set_context("talk")
    fig = sns.jointplot(y_true, y_predicted, kind="hex", space=0).set_axis_labels(
        "true", "pred"
    )
    plt.close(plt.gcf())
    return fig


def log_experiment(
    params={},
    metrics={},
    artifacts={},
    experiment_name="my_experiment",
    mlflow_tracking_uri="./experiments",
    mlflow_artifact_location=None,
):
    """
    Evaluate the model and log it with mlflow

    Args:
        params (dict): dictionary of parameters to log
        metrics (dict): dictionary of metrics to log
        artifacts (dict): dictionary of artifacts (path) to log
        experiment_name (str): experiment name
        mlflow_tracking_uri (str): path or sql url for mlflow logging
        mlflow_artifact_location (str): path or s3bucket url for artifact
            logging. If none, it will default to a standard.

    Returns:
        None
    """
    # Try to create an experiment if it doesn't exist
    try:
        exp_0 = mlflow.create_experiment(
            experiment_name, artifact_location=mlflow_artifact_location
        )
        # set uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Created new experiment id: {exp_0}")
    except Exception as E:
        logger.info(f"{E}. Writing to same URI/artifact store")
    # Always set the experiment
    mlflow.set_experiment(experiment_name)
    logger.info(f"Running experiment {experiment_name}")
    with mlflow.start_run():
        # param logging
        for key, val in params.items():
            logging.info(f"Logging param {key}")
            mlflow.log_param(key, val)
        # metric logging
        for key, val in metrics.items():
            logging.info(f"Logging metric {key}")
            mlflow.log_metric(key, val)
        # artifact logging
        for key, val in artifacts.items():
            logging.info(f"Logging artifact {key}")
            mlflow.log_artifact(val)


if __name__ == "__main__":
    setup_logging()
    load_dotenv(find_dotenv(), verbose=True, override=True)
    evaluate()
