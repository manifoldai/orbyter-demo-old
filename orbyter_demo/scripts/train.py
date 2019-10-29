# -*- coding: utf-8 -*-
"""
This module is for training models.

Examples:
    Example command line executable::

        $ python train.py /mnt/configs/config.yml
"""
import logging
from pathlib import Path

import click
import pandas as pd
from cloudpickle import dump
from sklearn.ensemble import RandomForestRegressor

from orbyter_demo.util.config import parse_config
from orbyter_demo.util.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True), default="/mnt/configs/config.yml"
)
def main(config_file):
    """
    Main function that trains a model and persists to disk.

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

    # Actually train the model
    logger.info("Training model.")
    model = RandomForestRegressor(n_estimators=10)
    trained_model = train(model, X, y)

    # Persist model
    logger.info(f"Persisting model to {config['model']['model_path']}.")
    model_path = Path(config["model"]["model_path"])
    with open(model_path, "wb") as f:
        dump(trained_model, f)


def train(model, X, y):
    """
    Model trainer for models and pipelines that adhere to the sklearn API.

    Args:
        model (sklearn.model): model that adheres to sklearn model API
        X (DataFrame): feature matrix
        y (DataFrame): target variable

    Returns:
        trained_model (sklearn.model): trained model
    """
    trained_model = model.fit(X, y)
    return trained_model


if __name__ == "__main__":
    setup_logging()
    main()
