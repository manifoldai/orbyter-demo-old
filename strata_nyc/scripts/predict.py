# -*- coding: utf-8 -*-
"""
This module is for running predictions.

Examples:
    Example command line executable::

        $ python predict.py
"""
import logging
from pathlib import Path

import click
import pandas as pd
from cloudpickle import load

from strata_nyc.util.config import parse_config
from strata_nyc.util.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument("config_file", type=str, default="/mnt/configs/config.yml")
def predict(config_file):
    """
    Main function runs predictions.

    Args:
        None

    Returns:
        None
    """
    config = parse_config(config_file)

    # Load model
    logger.info(f"Loading model from {config['predict']['model_path']}.")
    model_path = Path(config["predict"]["model_path"])
    with open(model_path, "rb") as f:
        trained_model = load(f)

    # Load data
    logger.info(f"Loading input data from {config['predict']['data_path']}.")
    data_path = Path(config["predict"]["data_path"])
    X = pd.read_parquet(data_path)

    # Make predictions and persist
    logger.info(
        f"Make predictions and persist to {config['predict']['predictions_path']}."
    )
    yhat = trained_model.predict(X)
    yhat = pd.DataFrame(yhat, columns=["MedianHouseValue"])
    yhat.to_parquet(config["predict"]["predictions_path"])


if __name__ == "__main__":
    setup_logging()
    predict()
