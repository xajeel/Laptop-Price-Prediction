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
# Purpose of Data Ingestion File is to read the Data from Data Source
# Split the Data in Train and Test datasets

# Input to our Data ingestion class will be given through this class 


class DataIngestion:

    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def data_ingestion(self):
        logging.info("Data Ingestion Method Starts")

        try:
        #We can read the data from any data source here at this line            
            df = pd.read_csv(r'notebook\data\laptop_data.csv')
            logging.info("Dataset read as pandas DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

        # saving the Raw Data 
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Ingestion of Data Completed')

            return (
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    ingestion_object = DataIngestion()
    ingestion_object.data_ingestion()

# Data Cleaning 
    final_obj = DataCleaning()
    final_obj.data_cleaning(final_obj.data_cleaning_config.raw_data)

# Data Transformation File 
    transform_object = DataTransformation()
    transform_object.Data_spliting()
    transform_object.processed_data()

# Model Trainer 

    model_object = ModelTrainer()
    data_transformation_object = DataTransformation()
    input_train, input_test = data_transformation_object.processed_data()
    model_object.model_trainer(x_train=input_train[:,:-1], y_train=input_train[:,-1], x_test=input_test[:,:-1], y_test=input_test[:,-1])

