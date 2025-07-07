import argparse
import os
from unittest.mock import Mock, patch

import pytest

from hfung4_dbr_ml_utils.common import (
    MLOpsParser,
    get_dbr_token,
    get_workspace_url,
)


class TestCommonFunctions:
    """Test cases for common utility functions."""

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "13.0"})
    @patch("hfung4_dbr_ml_utils.common.SparkSession")
    @patch("hfung4_dbr_ml_utils.common.DBUtils")
    def test_get_dbr_token_success(self, mock_dbutils_class, mock_spark_session):
        """Test getting Databricks token successfully."""
        mock_spark = Mock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark

        mock_dbutils = Mock()
        mock_dbutils_class.return_value = mock_dbutils
        mock_dbutils.notebook.entry_point.getDBUtils.return_value.notebook.return_value.getContext.return_value.apiToken.return_value.get.return_value = "test_token"

        token = get_dbr_token()

        assert token == "test_token"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_dbr_token_not_databricks(self):
        """Test getting token when not in Databricks environment."""
        with pytest.raises(ValueError, match="Not running in a Databricks environment"):
            get_dbr_token()

    @patch("hfung4_dbr_ml_utils.common.WorkspaceClient")
    def test_get_workspace_url(self, mock_workspace_client):
        """Test getting workspace URL."""
        mock_client = Mock()
        mock_client.config.host = "https://example.databricks.com"
        mock_workspace_client.return_value = mock_client

        url = get_workspace_url()

        assert url == "https://example.databricks.com"



class TestMLOpsParser:
    """Test cases for MLOpsParser class."""

    def test_mlops_parser_initialization(self):
        """Test MLOpsParser initializes correctly."""
        parser = MLOpsParser()
        
        # Should have a parser and subparsers
        assert hasattr(parser, 'parser')
        assert hasattr(parser, 'subparsers')
        assert hasattr(parser, 'common_args')

    def test_add_simple_command(self):
        """Test adding a simple command with common args only."""
        parser = MLOpsParser()
        parser.add_simple_command("data_ingestion", "Data ingestion options")
        
        # Test parsing with simple command
        result = parser.parse(["data_ingestion", "--root_path", "/test", "--env", "dev"])
        
        assert result.command == "data_ingestion"
        assert result.root_path == "/test"
        assert result.env == "dev"
        assert result.is_test == 0  # Default value

    def test_add_custom_command(self):
        """Test adding a custom command with additional args."""
        parser = MLOpsParser()
        parser.add_custom_command(
            "model_train_register",
            "Model training options",
            [
                {"--git_sha": {"type": str, "required": True, "help": "Git SHA"}},
                {"--job_run_id": {"type": str, "required": True, "help": "Job run ID"}},
                {"--branch": {"type": str, "required": True, "help": "Git branch"}},
            ]
        )
        
        test_args = [
            "model_train_register",
            "--root_path", "/test",
            "--env", "prod",
            "--is_test", "1",
            "--git_sha", "abc123",
            "--job_run_id", "456",
            "--branch", "main"
        ]
        
        result = parser.parse(test_args)
        
        assert result.command == "model_train_register"
        assert result.root_path == "/test"
        assert result.env == "prod"
        assert result.is_test == 1
        assert result.git_sha == "abc123"
        assert result.job_run_id == "456"
        assert result.branch == "main"

    def test_multiple_commands(self):
        """Test adding multiple commands."""
        parser = MLOpsParser()
        parser.add_simple_command("data_ingestion", "Data ingestion")
        parser.add_simple_command("deployment", "Deployment")
        parser.add_custom_command(
            "monitoring",
            "Monitoring options",
            [{"--alert_email": {"type": str, "required": True, "help": "Alert email"}}]
        )
        
        # Test first command
        result1 = parser.parse(["data_ingestion", "--root_path", "/test", "--env", "dev"])
        assert result1.command == "data_ingestion"
        
        # Test second command
        result2 = parser.parse(["deployment", "--root_path", "/test", "--env", "staging"])
        assert result2.command == "deployment"
        
        # Test custom command
        result3 = parser.parse(["monitoring", "--root_path", "/test", "--env", "prod", "--alert_email", "test@example.com"])
        assert result3.command == "monitoring"
        assert result3.alert_email == "test@example.com"

    def test_common_args_validation(self):
        """Test common arguments validation."""
        parser = MLOpsParser()
        parser.add_simple_command("test_cmd", "Test command")
        
        # Test required root_path
        with pytest.raises(SystemExit):
            parser.parse(["test_cmd", "--env", "dev"])
        
        # Test required env
        with pytest.raises(SystemExit):
            parser.parse(["test_cmd", "--root_path", "/test"])

    def test_is_test_default_value(self):
        """Test is_test has correct default value."""
        parser = MLOpsParser()
        parser.add_simple_command("test_cmd", "Test command")
        
        result = parser.parse(["test_cmd", "--root_path", "/test", "--env", "dev"])
        assert result.is_test == 0

    def test_is_test_explicit_value(self):
        """Test is_test with explicit value."""
        parser = MLOpsParser()
        parser.add_simple_command("test_cmd", "Test command")
        
        result = parser.parse(["test_cmd", "--root_path", "/test", "--env", "dev", "--is_test", "1"])
        assert result.is_test == 1

    def test_no_commands_fails(self):
        """Test that parser fails when no commands are added."""
        parser = MLOpsParser()
        
        with pytest.raises(SystemExit):
            parser.parse(["nonexistent_command", "--root_path", "/test", "--env", "dev"])

    def test_invalid_command_fails(self):
        """Test that parser fails with invalid command."""
        parser = MLOpsParser()
        parser.add_simple_command("valid_command", "Valid command")
        
        with pytest.raises(SystemExit):
            parser.parse(["invalid_command", "--root_path", "/test", "--env", "dev"])

    def test_help_text_optional(self):
        """Test that help text is optional for simple commands."""
        parser = MLOpsParser()
        parser.add_simple_command("test_cmd")  # No help text (uses default None)
        
        result = parser.parse(["test_cmd", "--root_path", "/test", "--env", "dev"])
        assert result.command == "test_cmd"
        
        # Also test with explicit None
        parser2 = MLOpsParser()
        parser2.add_simple_command("test_cmd2", None)
        
        result2 = parser2.parse(["test_cmd2", "--root_path", "/test", "--env", "dev"])
        assert result2.command == "test_cmd2"
