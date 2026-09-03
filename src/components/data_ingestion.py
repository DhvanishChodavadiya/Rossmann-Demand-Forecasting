import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

@dataclass

class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Started data ingestion")
            sales_data = pd.read_csv('Data/train.csv',parse_dates=['Date'])
            store_data = pd.read_csv('Data/store.csv')
            df = sales_data.merge(store_data, on='Store', how='left')
            logging.info('Combined sales and store data')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, header=True,index=False)
            logging.info('Saved raw data')

            df = df.sort_values('Date')

            cutoff_date = '2015-06-01'

            train_data = df[df['Date'] < cutoff_date]
            test_data = df[df['Date'] >= cutoff_date]

            # X_train = train.drop(['Sales', 'Date'], axis=1)
            # y_train = train['Sales']

            # X_test = valid.drop(['Sales', 'Date'], axis=1)  
            # y_test = valid['Sales']

            train_data.to_csv(self.data_ingestion_config.train_data_path, header=True, index=False)
            logging.info('Saved training data')
            test_data.to_csv(self.data_ingestion_config.test_data_path, header=True, index=False)
            logging.info('Saved testing data')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_data_path,test_data_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    data_transformation_obj.initiate_data_transformation(train_data_path,test_data_path)
