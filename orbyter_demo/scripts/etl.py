# -*- coding: utf-8 -*-
"""
This module is for the etl of raw data into processed data.

Examples:
    Example command line executable::

        $ python etl.py /mnt/configs/config.yml
"""
import logging
from pathlib import Path

import click
import pandas as pd
from sklearn import datasets

from orbyter_demo.util.config import parse_config
from orbyter_demo.util.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument("config_file", type=str, default="/mnt/configs/config.yml")
def etl(config_file):
    """
    ETL function that loads raw data, converts to df, and writes to processed dir.

    Args:
        config_file (str): path to config file

    Returns:
        None
    """
    # Parse relevant data from config file
    config = parse_config(config_file)
    raw_data_dir = Path(config["global"]["raw_data_dir"])
    processed_data_dir = Path(config["global"]["processed_data_dir"])

    # Load dataset and convert to pandas dataframe
    logger.info(f"Extract data from CA housing dataset in {raw_data_dir}.")
    data = datasets.fetch_california_housing(data_home=raw_data_dir)
    y = pd.DataFrame(data.target, columns=["MedianHouseValue"])
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Save as parquet file in processed dir
    logger.info(f"Write processed data to parquet file in {processed_data_dir}.")
    X.to_parquet(processed_data_dir / "X.pqt")
    y.to_parquet(processed_data_dir / "y.pqt")


if __name__ == "__main__":
    setup_logging()
    etl()
