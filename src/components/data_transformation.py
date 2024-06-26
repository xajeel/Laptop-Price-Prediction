import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from config import DataTrnasformationConfig


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrnasformationConfig()
    
    def Data_spliting(self):
        try:
        # Spliting Data 
            logging.info('Data Spliting Start')
            data = pd.read_csv(self.data_transformation_config.clean_data_path)
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)
            logging.info('Data Spliting Complete')
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def data_Transformation(self):
        """
        This Function Performs Data Transformation
        """
        try:
        # Extracting Numeric & Categorical Features 
            logging.info('Extracting Categorical & Numerical Data')
            df = pd.read_csv(self.data_transformation_config.clean_data_path)
            categorical = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys']
            numerical = ['Ram', 'ips', 'touch', 'screen_dpi', 'ssd', 'hdd']

        # Applying Data Transformation 
            logging.info('Data Transformation Start')
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('Encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical),
                ('cat_pipeline', cat_pipeline, categorical)
            ])

            logging.info('Data Transformation Complete')
            logging.info('Saving Preprocessed')
            path = self.data_transformation_config.preprocess_path
            save_object(save_path=path, best_model=preprocessor)

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def processed_data(self):
        preproc = self.data_Transformation()
        train = pd.read_csv(self.data_transformation_config.train_data_path)
        test = pd.read_csv(self.data_transformation_config.test_data_path)

        x_train = train.drop(columns=['Price'])
        y_train = train['Price']

        x_test = test.drop(columns=['Price'])
        y_test = test['Price']

        # Transforming the Data 

        input_train_arr = preproc.fit_transform(x_train)
        input_test_arr = preproc.transform(x_test)

        # Combbing the input and Output 
        input_train = np.c_[input_train_arr, np.array(np.log(y_train))]
        input_test = np.c_[input_test_arr, np.array(np.log(y_test))]

        return (
            input_train,
            input_test
        )



            