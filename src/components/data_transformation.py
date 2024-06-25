import os
import sys
from data_cleaning import DataCleaningConfig


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataTrnasformationConfig:
    data_config = DataCleaningConfig()
    clean_data_path = data_config.clean_data_path
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrnasformationConfig()
    
    def Data_spliting(self):
    # Spliting Data 
        train_set, test_set = train_test_split(self.data_transformation_config.clean_data_path, test_size=0.2, random_state=42)
        train_set.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)