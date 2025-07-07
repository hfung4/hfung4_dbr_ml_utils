"""
Common functions used across projects.
"""

import argparse
import os

from collections.abc import Sequence
from typing import List, Dict, Any

from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


def is_databricks() -> bool:
    """
    Check if the code is running in a Databricks environment.

    Returns:
        bool: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_dbr_token() -> str:
    """
    Get the Databricks API token from the notebook context.

    Returns:
        str: The Databricks API token as a string if available, otherwise an empty string.

    Raises:
        ValueError: If not running in a Databricks environment or if the token is not found.
    """
    if is_databricks():
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        return (
            dbutils.notebook.entry_point.getDBUtils()
            .notebook()
            .getContext()
            .apiToken()
            .get()
        )
    else:
        raise ValueError("Not running in a Databricks environment or token not found.")


def get_workspace_url() -> str:
    """
    Retrieve the Databricks workspace URL.

    Returns:
        str: Databricks workspace URL as a string.
    """
    w = WorkspaceClient()
    return w.config.host


class MLOpsParser:
    """
    A flexible argument parser for MLOPs operations in Databricks.

    This class provides a structured way to create argument parsers with common arguments
    and allows for dynamic addition of subcommands with custom arguments.

    Example (providing args explicitly):
        >>> from hfung4_dbr_ml_utils.common import MLOpsParser
        >>> parser = MLOpsParser()
        >>> parser.add_simple_command(name="data_validation", help_text="Data validation options")
        >>> args = parser.parse(["data_validation", "--root_path", "/path", "--env", "dev"])

    Example (uses sys.argv from terminal or DAB automatically):
        >>> from hfung4_dbr_ml_utils.common import MLOpsParser
        >>> parser = MLOpsParser()
        >>> parser.add_simple_command("data_ingestion", "Data ingestion options")
        >>> # Terminal: python script.py data_ingestion --root_path /path --env dev
        >>> # DAB: parameters defined in databricks.yml get passed automatically
        >>> args = parser.parse()
        >>> root_path = args.root_path
        >>> env = args.env
        >>> command = args.command # command will be 'data_ingestion'
    """

    def __init__(self):
        """Initialize the MLOps parser with common arguments."""
        self.parser = argparse.ArgumentParser(
            description="Parser for MLOPs in Databricks"
        )
        self.subparsers = self.parser.add_subparsers(dest="command", required=True)
        self.common_args = self._create_common_args()

    def _create_common_args(self) -> argparse.ArgumentParser:
        """Create common arguments used across all MLOPs workflows."""
        common_args = argparse.ArgumentParser(add_help=False)
        common_args.add_argument(
            "--root_path",
            type=str,
            required=True,
            help="Root path for the MLOPs project in DAB",
        )
        common_args.add_argument(
            "--env",
            type=str,
            required=True,
            help="Target environment e.g., dev, staging, prod",
        )
        common_args.add_argument(
            "--is_test",
            type=int,
            required=False,
            default=0,
            help="Flag to indicate if integration test is running test run (1) or not (0)",
        )
        return common_args

    def add_simple_command(self, name: str, help_text: str = None):
        """
        Add a command with only common arguments.

        Args:
            name (str): Name of the command
            help_text (str): Help text for the command
        """
        self.subparsers.add_parser(name, parents=[self.common_args], help=help_text)

    def add_custom_command(
        self, name: str, help_text: str, additional_args: List[Dict[str, Any]]
    ):
        """
        Add a command with common arguments plus custom arguments.

        Args:
            name (str): Name of the command
            help_text (str): Help text for the command
            additional_args (List[Dict[str, Any]]): List of dictionaries containing argument definitions
                The list of dictionaries should have the format:
                [{"--arg_name": {"type": type, "required": bool, "default": default_value, "help": str}, ...]
        """
        cmd_parser = self.subparsers.add_parser(
            name, parents=[self.common_args], help=help_text
        )
        for arg_dict in additional_args:
            for arg_name, arg_config in arg_dict.items():
                cmd_parser.add_argument(arg_name, **arg_config)

    def parse(self, args: Sequence[str] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Args:
            args (Sequence[str], optional): List of command-line arguments to parse.

        Returns:
            argparse.Namespace: Parsed command-line arguments.
        """
        return self.parser.parse_args(args)
