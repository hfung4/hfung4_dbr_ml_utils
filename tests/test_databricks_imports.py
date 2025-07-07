"""Test databricks dependencies can be imported without errors."""

import pytest


def test_databricks_connect_import():
    """Test that databricks-connect can be imported without protobuf errors."""
    try:
        from databricks.connect import DatabricksSession
        # If we get here, the import succeeded
        assert True
    except ImportError as e:
        if "runtime_version" in str(e) and "protobuf" in str(e):
            pytest.fail(f"Protobuf compatibility issue: {e}")
        else:
            # Re-raise other import errors as they might be expected
            raise


def test_protobuf_runtime_version():
    """Test that protobuf has the required runtime_version attribute."""
    import google.protobuf
    
    # Check if runtime_version exists
    assert hasattr(google.protobuf, 'runtime_version'), "protobuf missing runtime_version - version incompatibility"