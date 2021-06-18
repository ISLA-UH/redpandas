# RedPandas
from redpandas.redpd_dw_to_parquet import redpd_dw_to_parquet

if __name__ == "__main__":
    """
    Extract RedVox data into a Pandas DataFrame
    """
    # Absolute path
    INPUT_DIR = "path/to/redvox/data"

    redpd_dw_to_parquet(input_dir=INPUT_DIR)
