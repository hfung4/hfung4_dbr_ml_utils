from unittest.mock import Mock, patch

from pyspark.sql import SparkSession

from hfung4_dbr_ml_utils.delta import get_delta_table_version


class TestDeltaFunctions:
    """Test cases for Delta table utility functions."""

    @patch("hfung4_dbr_ml_utils.delta.DeltaTable")
    def test_get_delta_table_version_success(self, mock_delta_table):
        """Test successful retrieval of Delta table version."""
        # Mock the Delta table and its methods
        mock_table = Mock()
        mock_history = Mock()
        mock_history.select.return_value.first.return_value = [5]
        mock_table.history.return_value = mock_history
        mock_delta_table.forName.return_value = mock_table

        mock_spark = Mock(spec=SparkSession)

        result = get_delta_table_version("catalog", "schema", "table", mock_spark)

        assert result == 5
        mock_delta_table.forName.assert_called_once_with(mock_spark, "catalog.schema.table")
        mock_table.history.assert_called_once()
        mock_history.select.assert_called_once_with("version")

    @patch("hfung4_dbr_ml_utils.delta.DeltaTable")
    def test_get_delta_table_version_with_default_spark(self, mock_delta_table):
        """Test Delta table version retrieval with default Spark session."""
        mock_table = Mock()
        mock_history = Mock()
        mock_history.select.return_value.first.return_value = [3]
        mock_table.history.return_value = mock_history
        mock_delta_table.forName.return_value = mock_table

        with patch("hfung4_dbr_ml_utils.delta.SparkSession") as mock_spark_session:
            mock_spark = Mock()
            mock_spark_session.builder.getOrCreate.return_value = mock_spark

            result = get_delta_table_version("catalog", "schema", "table")

            assert result == 3
            mock_spark_session.builder.getOrCreate.assert_called_once()
            mock_delta_table.forName.assert_called_once_with(mock_spark, "catalog.schema.table")

    @patch("hfung4_dbr_ml_utils.delta.DeltaTable")
    def test_get_delta_table_version_value_error(self, mock_delta_table):
        """Test handling of ValueError during version retrieval."""
        mock_delta_table.forName.side_effect = ValueError("Table not found")

        mock_spark = Mock(spec=SparkSession)

        result = get_delta_table_version("catalog", "schema", "table", mock_spark)

        assert result is None

    @patch("hfung4_dbr_ml_utils.delta.DeltaTable")
    def test_get_delta_table_version_runtime_error(self, mock_delta_table):
        """Test handling of RuntimeError during version retrieval."""
        mock_delta_table.forName.side_effect = RuntimeError("Connection failed")

        mock_spark = Mock(spec=SparkSession)

        result = get_delta_table_version("catalog", "schema", "table", mock_spark)

        assert result is None

    @patch("hfung4_dbr_ml_utils.delta.DeltaTable")
    def test_get_delta_table_version_table_path_format(self, mock_delta_table):
        """Test that table path is formatted correctly."""
        mock_table = Mock()
        mock_history = Mock()
        mock_history.select.return_value.first.return_value = [1]
        mock_table.history.return_value = mock_history
        mock_delta_table.forName.return_value = mock_table

        mock_spark = Mock(spec=SparkSession)

        get_delta_table_version("test_catalog", "test_schema", "test_table", mock_spark)

        mock_delta_table.forName.assert_called_once_with(mock_spark, "test_catalog.test_schema.test_table")

    @patch("hfung4_dbr_ml_utils.delta.DeltaTable")
    @patch("builtins.print")
    def test_get_delta_table_version_error_logging(self, mock_print, mock_delta_table):
        """Test that errors are properly logged."""
        error_message = "Test error message"
        mock_delta_table.forName.side_effect = ValueError(error_message)

        mock_spark = Mock(spec=SparkSession)

        result = get_delta_table_version("catalog", "schema", "table", mock_spark)

        assert result is None
        mock_print.assert_called_once_with(f"Error retrieving Delta table version: {error_message}")
