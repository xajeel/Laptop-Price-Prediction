import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Purpose of Data Ingestion File is to read the Data from Data Source
# Split the Data in Train and Test datasets

# Input to our Data ingestion class will be given through this class 
@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')


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

        #     logging.info('Train Test Split')
        #     train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

        # # saving the training and testing CSV files 
        #     train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        #     test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of Data Completed')

            return (
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    ingestion_object = DataIngestion()
    ingestion_object.data_ingestion()