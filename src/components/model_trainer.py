import sys
import os
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from src.constant import *
from src.exception import CustumExceptions
from src.utils import main_utils
from src.logger import logging

from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, 'model.pkl')
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('Config', 'model.yaml')


class ModelTrainer:
    def __init__(self):
        
        self.model_training_config = ModelTrainingConfig()
        self.utils = main_utils()

        self.model = {
            'XGBClassifier' : XGBClassifier(),
            'GradientBoostingClassifier' : GradientBoostingClassifier(),
            'SVC' : SVC(),
            'RandomForestClassifier' : RandomForestClassifier()
        } 

    
    def evaluate_models(self, X, y, models):
        try:
            X_train,X_test,y_train,y_test = train_test_split(
                X,y, test_size=0.2, random_state=42
            )

            report = {}

            for i in range(len(list(models))):
                model = list(models.value())[i]

                model.fit(X_train,y_train)

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)

                test_model_score = accuracy_score(y_test,y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report
        
        except Exception as e:
            raise CustumExceptions(e, sys)
        
    
    def get_best_model(self,
                       X_train: np.array,
                       y_train: np.array,
                       X_test: np.array,
                       y_test: np.array):
        
        try:

            model_report : dict = self.evaluate_models(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models= self.models
            )

            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            ## to get the best_model_name from dict 

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score
        
        except Exception as e:
            raise CustumExceptions(e, sys)
        
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train
                            ) -> object:
        
        try:

            model_param_grid = self.utils.read_yaml_file(self.model_training_config.model_config_file_path)['model_selection']['model'][best_model_name]['search_param_grid']


            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train,y_train)

            best_param = grid_search.best_params_

            print('best params are:', best_param)

            finetuneed_model = best_model_object.set_params(**best_param)

            return finetuneed_model
        
        except Exception as e:
            raise CustumExceptions(e,sys)


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f'splitting training and testing input and target feature')

            X_train, X_test, y_train, y_test = (
                train_array[:,:-1],
                train_array[:,:-1],
                test_array[:,:-1],
                test_array[:,:-1]
                
            )

            logging.info(f'Extracting model config file path')

            model_report : dict = self.evaluate_models(X=X_train, y=y_train, models=self.models)

            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = self.models[best_model_name]


            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train= X_train,
                y_train = y_train
            )

            best_model.fit(X_train,y_train)
            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f'best model name {best_model_name} ans score: {best_model_score}')

            if best_model_score < 0.5:
                raise Exception('no best model found witn an accuracy greater than the threashold 0.6')

            logging.info(f'Best model found on both trainig and testing dataset')


            logging.info(f'saving model as path:{self.model_training_config.trained_model_path}')

            os.makedirs(os.path.dirname(self.model_training_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path = self.model_training_config.trained_model_path,
                obj = best_model
            )

            return self.model_training_config.trained_model_path


        except Exception as e:
            raise CustumExceptions(e, sys) 
                  

            

        


