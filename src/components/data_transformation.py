import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import imputing_missing_values

@dataclass

class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation_object(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_df_path,test_df_path):
        try:
            train_df = pd.read_csv(train_df_path)
            test_df = pd.read_csv(test_df_path)

            imputed_train_df,imputed_test_df = imputing_missing_values(train_df,test_df)
            logging.info('Successfully imputed all missing values')
        except Exception as e:
            raise CustomException(e,sys)