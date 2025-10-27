import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Still use LabelEncoder first
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import sys
import os

from src.utils.common import read_yaml
from src.utils.logger import logger



def main():
    """
    Loads train/test data, defines and fits the ColumnTransformer preprocessor,
    encodes target labels appropriately for Keras multi-output classification,
    and saves the preprocessor and target encoders/mappings.
    """
    try:

        config = read_yaml(Path("configs/config.yaml"))
        params = read_yaml(Path("configs/params.yaml"))
        train_data_path = Path(config.data_paths.train_data)
        test_data_path = Path(config.data_paths.test_data)
        
        logger.info(" Starting Feature Engineering ")

        # check if exists
        if not train_data_path.exists() or not test_data_path.exists():
            logger.error(f"Train ({train_data_path}) or Test ({test_data_path}) data file not found.")
            sys.exit(1)

        # load data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path) 
        logger.info(f"Loaded training data Shape: {train_df.shape}, Loaded test data Shape: {test_df.shape}")

        # Define feature columns
        text_feature = params.features.combined_text_col
        categorical_features = params.features.categorical_features
        target_cols = params.features.targets 

        # just checking
        logger.info(f"Text feature: {text_feature}, Categorical features: {categorical_features}, Target columns: {target_cols}")

        # target encoder saving path
        encoder_dir = Path(config.model_paths.queue_encoder).parent
        os.makedirs(encoder_dir, exist_ok=True) # ensure directory exists

        # ^^^^^Encode Target Columns^^^^^
        logger.info("Processing target columns for Keras...")
        for target in target_cols:  
            le = LabelEncoder()
            le.fit(train_df[target])
            encoder_filename = f"{target}_encoder.pkl"
            encoder_save_path = encoder_dir / encoder_filename
            joblib.dump(le, encoder_save_path)
            logger.info(f"Saved LabelEncoder for '{target}' to {encoder_save_path}")


        logger.info("Defining feature preprocessing pipelines...")
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=None, # Already cleaned
                max_features=10000,
                ngram_range=(1,2)
                ))
        ])

        categorical_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ])

        # --- Create ColumnTransformer ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_pipeline, text_feature),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='drop' # i guess this is default😊
        )

        # drop target columns so we only fit on input features
        X_train = train_df.drop(columns=target_cols)        

        # --- Fit the Preprocessor ---
        logger.info(f"Fitting the preprocessor on training data features (Shape: {X_train.shape})...")
        preprocessor.fit(X_train)
        logger.info("Preprocessor fitting complete.")

        # --- Save the Fitted Preprocessor ---
        preprocessor_save_path = Path(config.model_paths.preprocessor)
        preprocessor_save_path.parent.mkdir(parents=True, exist_ok=True) # ensure dir exists
        joblib.dump(preprocessor, preprocessor_save_path)
        logger.info(f"Saved fitted preprocessor to: {preprocessor_save_path}")

        # i think this is not correct time to transform and show shape
        # but just to give an idea 
        # X_train_transformed = preprocessor.transform(X_train)
        
        logger.info("--- Feature Engineering Completed Successfully ---")

    except FileNotFoundError:
        logger.error(f"Error: A required data file was not found during processing.")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Error accessing configuration key: {e}. Check config.yaml and params.yaml structure.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during feature engineering: {e}")
        raise e

if __name__ == "__main__":
    main()