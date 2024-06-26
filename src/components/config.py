from dataclasses import dataclass
import os
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')

@dataclass
class DataCleaningConfig:
        raw_data = os.path.join('artifacts', 'raw.csv')
        clean_data_path = os.path.join('artifacts', 'clean_data.csv')

@dataclass
class DataTrnasformationConfig:
    logging.info('Data Paths')
    clean_data_path = os.path.join('artifacts', 'clean_data.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    preprocess_path = os.path.join('artifacts', 'preprocess.pkl')

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')