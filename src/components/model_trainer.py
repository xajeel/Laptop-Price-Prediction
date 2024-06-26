import os
import sys
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object
from config import ModelTrainerConfig

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_trainer(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=2),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=90,random_state=5)
            }
            
            report = {}
            for model_name, model in models.items():
                model.fit(x_train, y_train)
                y_predict = model.predict(x_test)
                test_model_score = r2_score(y_test, y_predict)
                report[model_name] = test_model_score
            
            rscore = max(report.values())
            model_name = list(report.keys())[list(report.values()).index(rscore)]
            best_model = models[model_name]
            model_save_path = self.model_trainer_config.model_path
            save_object(save_path=model_save_path, best_model=best_model)

            if rscore < 0.6:
                raise CustomException('No Model Found with r2 SCore greater than 0.6')

        except Exception as e:
            raise CustomException(e, sys)

