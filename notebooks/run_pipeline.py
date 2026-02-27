from factor_model.logging_utils import setup_logging
from factor_model.pipeline import run_quarterly

if __name__ == "__main__":
    setup_logging()
    run_quarterly(
        overall_start="2023-01-01",
        overall_end=None,  # infer from prices.parquet
        specific_var_method="total_minus_explained",
    )
