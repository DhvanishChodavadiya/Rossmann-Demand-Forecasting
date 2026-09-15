import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import imputing_missing_values
from src.utils import feature_engineering
from src.utils import save_object


@dataclass

class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.num_feature = ['CompetitionDistance','CompetitionOpen','Promo2OpenSinceMonths','Store_avg_sales','Store_avg_customers']
        self.cat_feature = ['Year','StoreType','Assortment']

    def data_transformation_object(self):
        try:
            preprocessor = ColumnTransformer([
                        ('scaler',StandardScaler(),self.num_feature),
                        ('ohe',OneHotEncoder(drop='first',sparse_output=True),self.cat_feature)
                    ],remainder='passthrough')

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_df_path,test_df_path):
        try:
            train_df = pd.read_csv(train_df_path)
            test_df = pd.read_csv(test_df_path)

            imputed_train_df,imputed_test_df = imputing_missing_values(train_df,test_df)
            logging.info('Successfully imputed all missing values')

            train_df_fe,test_df_fe = feature_engineering(imputed_train_df,imputed_test_df)
            logging.info("Feature engineering completed")

            X_train = train_df_fe.drop(['Sales', 'Date'], axis=1)
            y_train = train_df_fe['Sales']

            X_test = test_df_fe.drop(['Sales', 'Date'], axis=1)
            y_test = test_df_fe['Sales'] 

            logging.info('Data are splitted into X_train,X_test,y_train,y_test')

            preprocessor_obj = self.data_transformation_object()
            logging.info("Got preprocessor object")

            X_train = preprocessor_obj.fit_transform(X_train)
            X_test = preprocessor_obj.transform(X_test)

            y_train = np.log1p(y_train)
            logging.info("Data preprocessed successfully")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            logging.info("Saved Preprocessor successfully")

            return X_train,X_test,y_train,y_test

        except Exception as e:
            raise CustomException(e,sys)