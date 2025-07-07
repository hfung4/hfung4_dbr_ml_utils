# hfung4_dbr_ml_utils

Common utilities for Databricks ML projects.

## Project Structure

```
hfung4_dbr_ml_utils/
|-- src/
|   +-- hfung4_dbr_ml_utils/
|       |-- __init__.py          # Package initialization and exports
|       |-- common.py            # Databricks utilities and MLOps parser
|       +-- delta.py             # Delta table utilities
|-- tests/
|   |-- test_common.py           # Tests for common utilities
|   |-- test_databricks_imports.py
|   +-- test_delta.py            # Tests for Delta table functions
|-- pyproject.toml               # Project configuration and dependencies
|-- Taskfile.yaml                # Task runner configuration
|-- databricks.yml               # Databricks bundle configuration
|-- uv.lock                      # Dependency lock file
+-- version.txt                  # Package version
```

## Installation

```bash
pip install hfung4_dbr_ml_utils
```

## Usage

### Common Utilities

```python
from hfung4_dbr_ml_utils import is_databricks, get_dbr_token, get_workspace_url

# Check if running in Databricks
if is_databricks():
    token = get_dbr_token()
    workspace_url = get_workspace_url()
```

### MLOps Argument Parser

```python
from hfung4_dbr_ml_utils import MLOpsParser

parser = MLOpsParser()
parser.add_simple_command("data_ingestion", "Data ingestion options")
args = parser.parse()
```

### Delta Table Utilities

```python
from hfung4_dbr_ml_utils import get_delta_table_version

version = get_delta_table_version("catalog", "schema", "table")
```

## Development

### Setup

```bash
# Create virtual environment and install dependencies
task sync-dev
```

### Commands

```bash
# Run tests
task run-unit-tests

# Run linting
task lint

# Build package
task build

# Clean artifacts
task clean
```

## License

MIT
