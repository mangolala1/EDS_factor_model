from factor_model.logging_utils import setup_logging
from factor_model.bulk_download import download_all_data

if __name__ == "__main__":
    setup_logging()
    download_all_data(start_date="2020-01-01", end_date=None, data_dir="data")
