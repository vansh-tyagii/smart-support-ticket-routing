import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.utils.common import read_yaml
from src.utils.logger import logger
import sys 

def main():

    try:    
        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))

        logger.info("Starting data ingestion process.")

        source_data_path = Path(config.data_paths.source_data)
        train_data_path = Path(config.data_paths.train_data)
        test_data_path = Path(config.data_paths.test_data)

        test_size = params.data_split.test_size
        random_state = params.data_split.random_state
        stratify_col = params.features.stratify[0]

        # check if source data file exists
        if not source_data_path.exists():
            logger.error(f"Source data file not found at: {source_data_path}")
            sys.exit(1)
        # load data
        df = pd.read_csv(source_data_path)
        logger.info(f"Source data loaded successfully from: {source_data_path}, shape: {df.shape}")

        df_columns = df.columns.tolist()
        logger.info(f"Data columns: {df_columns}")
        # check if stratify column exists in data
        if stratify_col not in df_columns:
            logger.error(f"Stratify column '{stratify_col}' not found in data columns.")
            sys.exit(1)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_col]
        )
    
        logger.info(f"Split data. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        logger.info(f"Train target distribution:\n{train_df[stratify_col].value_counts(normalize=True)}")
        logger.info(f"Test target distribution:\n{test_df[stratify_col].value_counts(normalize=True)}")

        # save the split data
        train_df.to_csv(train_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)
        logger.info(f"Train data saved to: {train_data_path}")
        logger.info(f"Test data saved to: {test_data_path}")
        logger.info("Data ingestion process completed successfully.")

    except FileNotFoundError:
         # This case should be caught earlier, but added for robustness
        logger.error(f"Error: A required file was not found during processing.")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Error accessing configuration key: {e}. Check config.yaml and params.yaml.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during data ingestion: {e}") # Log full traceback
        raise e # Re-raise the exception after logging

if __name__ == "__main__":
    main()