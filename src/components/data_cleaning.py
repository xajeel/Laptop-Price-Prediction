import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import re
from data_ingestion import DataIngestionConfig

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataCleaningConfig:
        data_paths = DataIngestionConfig()
        raw_data = data_paths.raw_data_path
        clean_data_path = os.path.join('artifacts', 'clean_data.csv')


class DataCleaning:
    def __init__(self):
        self.data_cleaning_config = DataCleaningConfig()

    def data_cleaning(self, data_path):
        """
        This Function is Responsible for Data Cleaning
        """

        try:

            ips = []
            touch = []
            x_resolution = []
            y_resolution = []
            cpu = []

        # Data Transformation 
            logging.info('Reading Data for Transformation')
            df = pd.read_csv(data_path)
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # Data Cleaning 
            logging.info('Data Cleaning Start')
        
        # 1 Removing Duplicates
            df = df.drop_duplicates()

        # 2 Cleaning Columns
            df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
            df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

        # Cleaning the Screen Resolution Column
            for s in df['ScreenResolution']:
                match = re.search(r'^(.*?)(\d+x\d+)$', s)
                value2 = match.group(1)
                value3  = match.group(2)
                
                if "IPS" in value2:
                    ips.append(1)
                else:
                    ips.append(0)

                if 'Touchscreen' in value2:
                    touch.append(1)
                else:
                    touch.append(0)

                x_resolution.append(int(value3.split('x')[0]))
                y_resolution.append(int(value3.split('x')[1]))

            df['ips'] = ips
            df['touch'] = touch
            df['x_resolution'] = x_resolution
            df['y_resolution'] = y_resolution


        # Droping  Screen Resolution Column 
            df.drop('ScreenResolution', axis=1, inplace=True)

        # Calculating DPI for Screen 
            df['screen_dpi'] = ((df['x_resolution'])**2 + (df['y_resolution'])**2)**0.5 / df['Inches']

        # Droping Resolution
            df.drop('x_resolution', axis=1, inplace=True)
            df.drop('y_resolution', axis=1, inplace=True)
            df.drop('Inches', axis=1, inplace=True)

        # cleaning CPu Column 
            for s in df['Cpu']:
                if 'i5' in s:
                    cpu.append(" ".join(s.split(' ')[:3]))
                elif 'i7' in s:
                    cpu.append(" ".join(s.split(' ')[:3]))
                elif 'i3' in s:
                    cpu.append(" ".join(s.split(' ')[:3]))
                elif 'AMD' in s:
                    cpu.append('AMD Processor')
                else:
                    cpu.append('other')
            df['Cpu'] = cpu
        

        # Cleaning Memory Column 
            def parse_storage(storage_str):
        # Parse SSD, HDD, Flash, Hybrid
                ssd = re.findall(r'(\d+\.?\d*\s*(?:GB|TB)\s*SSD)', storage_str, re.IGNORECASE)
                hdd = re.findall(r'(\d+\.?\d*\s*(?:GB|TB)\s*HDD)', storage_str, re.IGNORECASE)
                flash = re.findall(r'(\d+\.?\d*\s*(?:GB|TB)\s*Flash Storage)', storage_str, re.IGNORECASE)
                hybrid = re.findall(r'(\d+\.?\d*\s*(?:GB|TB)\s*Hybrid)', storage_str, re.IGNORECASE)
                
                # Convert to number (assume GB for simplicity)
                def extract_size(s):
                    num = float(re.match(r"(\d+\.?\d*)", s).group(1))
                    if "TB" in s:
                        num *= 1024  # Convert TB to GB
                    return num
                
                return {
                    "ssd": extract_size(ssd[0]) if ssd else 0,
                    "hdd": extract_size(hdd[0]) if hdd else 0,
                    "flash": extract_size(flash[0]) if flash else 0,
                    "hybrid": extract_size(hybrid[0]) if hybrid else 0,
                }

            # Parse all data
            parsed_data = [parse_storage(item) for item in df['Memory']]

            # Convert to DataFrame
            df2 = pd.DataFrame(parsed_data)

            # Show the DataFrame
            df = pd.concat([df, df2], axis=1)

        # Droping Memory Column
            df.drop(columns=['Memory'], inplace=True)
        
        # cleaning the GPU Column 

            df['Gpu'] = df['Gpu'].apply(lambda x: x.split(' ')[0])
            df = df[df['Gpu'] != 'ARM']

        # Cleaning the Operating System Column 

            for s in df['OpSys']:
                if 'Windows' in s:
                    df['OpSys'] = df['OpSys'].replace(s, 'Windows')
                elif 'Mac' in s:
                    df['OpSys'] = df['OpSys'].replace(s, 'Mac')
                else:
                    df['OpSys'] = df['OpSys'].replace(s, 'Others')
            
            
            df.drop(columns=['hybrid', 'flash'], inplace=True)
            df.to_csv(self.data_cleaning_config.clean_data_path, index=False, header=True)

            return self.data_cleaning_config.clean_data_path

        except Exception as e:
            raise CustomException(e,sys)
        
    


if __name__ == '__main__':
    final_obj = DataCleaning()
    final_obj.data_cleaning(final_obj.data_cleaning_config.raw_data)