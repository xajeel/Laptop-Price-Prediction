import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from config import DataIngestionConfig

from data_cleaning import DataCleaning
from data_transformation import DataTransformation
from model_trainer import ModelTrainer

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def data_ingestion(self):
        logging.info("Data Ingestion Method Starts")
        try:
            # Read the data from any data source here at this line            
            df = pd.read_csv(r'notebook\data\laptop_data.csv')
            logging.info("Dataset read as pandas DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Saving the Raw Data 
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Ingestion of Data Completed')

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        # Data Ingestion
        ingestion_object = DataIngestion()
        raw_data_path = ingestion_object.data_ingestion()

        # Data Cleaning
        final_obj = DataCleaning()
        final_obj.data_cleaning(raw_data_path)
        logging.info('Data Cleaning Completed')

        # Data Transformation
        transform_object = DataTransformation()
        transform_object.Data_spliting()
        transform_object.processed_data()
        logging.info('Data Transformation Completed')

        # Model Training
        model_object = ModelTrainer()
        input_train, input_test = transform_object.processed_data()
        model_object.model_trainer(
            x_train=input_train[:, :-1], 
            y_train=input_train[:, -1], 
            x_test=input_test[:, :-1], 
            y_test=input_test[:, -1]
        )
        logging.info('Model Training Completed')

    except Exception as e:
        logging.error("Error in main execution: %s", str(e))
        raise CustomException(e, sys)
