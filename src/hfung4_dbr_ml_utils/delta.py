"""
Delta table related functions.
"""

from delta.tables import DeltaTable
from pyspark.sql import SparkSession


def get_delta_table_version(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    spark: SparkSession | None = None,
) -> int:
    """Get the version of a Delta table.

    This function retrieves the version of a Delta table specified by its catalog, schema, and table name.

    Args:
        catalog_name (str): The name of the catalog where the Delta table is located.
        schema_name (str): The name of the schema where the Delta table is located.
        table_name (str): The name of the Delta table.
        spark (SparkSession, optional): The Spark session to use. If None, the current Spark session is used.

    Returns:
        int | None: The version of the Delta table, or None if error occurs
    """
    # Create or get existing Spark session
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    # Full table name in the format "catalog.schema.table"
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    try:
        # Get the Delta table
        delta_table = DeltaTable.forName(spark, full_table_name)
        # Get the version of the Delta table
        version = delta_table.history().select("version").first()[0]
        return version

    except (ValueError, RuntimeError) as e:
        print(f"Error retrieving Delta table version: {e}")
        return None
