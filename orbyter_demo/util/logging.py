# -*- coding: utf-8 -*-
"""
This module is for logging utility functions.
"""
import logging.config
import os

import coloredlogs
import yaml


def setup_logging(logging_config="logging.yml", default_level=logging.INFO):
    """
    Setup logging configuration

    Args:
        logging (str): logging yaml config
        default_level (logging.LEVEL): default logging level

    Returns:
        None
    """
    # from config
    if os.path.exists(logging_config):
        with open(logging_config, "rt") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
        config_method = logging_config
        # set colored log (console streaming) to use params set in config
        console_format = config["formatters"][
            config["handlers"]["console"]["formatter"]
        ]["format"]
        console_level = config["handlers"]["console"]["level"]
        console_stream = config["handlers"]["console"]["stream"]
        coloredlogs.install(fmt=console_format, level=console_level, sys=console_stream)
    # from default
    else:
        logging.basicConfig(level=default_level)
        config_method = "default_level"
        coloredlogs.install(level="DEBUG")
    logging.info(f"Logging set from {config_method}")
