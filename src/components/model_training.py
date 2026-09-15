import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import model_training
from src.utils import save_object

@dataclass

class ModelTrainingConfig:
    trained_model_path: str= os.path.join('artifacts','model.pkl') 

class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self,X_train,X_test,y_train,y_test):
        try:
            rmspe,model = model_training(X_train,X_test,y_train,y_test)
            logging.info("Training completed successfully")

            save_object(
                file_path=self.model_training_config.trained_model_path,
                obj=model
            )

            return rmspe

        except Exception as e:
            raise CustomException(e,sys)
