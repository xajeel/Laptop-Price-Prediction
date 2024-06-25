import os
import sys
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
import pickle as pk
from src.exception import CustomException

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')

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
            
            rscore = max(sorted(report.values()))
            if rscore < 0.6:
                raise CustomException('No Model Found with r2 SCore greater than 0.6')
            else:
                return rscore

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    model_object = ModelTrainer()
    data_transformation_object = DataTransformation()
    input_train, input_test = data_transformation_object.processed_data()
    model_output = model_object.model_trainer(x_train=input_train[:,:-1], y_train=input_train[:,-1], x_test=input_test[:,:-1], y_test=input_test[:,-1])

    print(model_output)


