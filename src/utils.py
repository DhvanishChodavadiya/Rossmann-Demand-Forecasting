import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def imputing_missing_values(train_df,test_df):
    try:
        logging.info("Starting imputing missing values")

        train_df['CompetitionDistance'].fillna(0,inplace=True)
        test_df['CompetitionDistance'].fillna(0,inplace=True)
        logging.info('Imputed CompetitionDistance')

        train_df['Date'] = pd.to_datetime(train_df['Date'])
        train_df['Year'] = train_df['Date'].dt.year
        train_df['Month'] = train_df['Date'].dt.month
        train_df['CompetitionOpen_missing'] = train_df['CompetitionOpenSinceYear'].isna().astype(int)
        train_df['CompetitionOpenSinceMonth'].fillna(1,inplace=True)
        train_df['CompetitionOpenSinceYear'].fillna(train_df['Year'].min(),inplace=True)
        train_df['CompetitionOpen'] = 12 * (train_df['Year'] - train_df['CompetitionOpenSinceYear']) + (train_df['Month'] - train_df['CompetitionOpenSinceMonth'])
        train_df['CompetitionOpen'] = train_df['CompetitionOpen'].apply(lambda x: max(x, 0))

        test_df['Date'] = pd.to_datetime(test_df['Date'])
        test_df['Year'] = test_df['Date'].dt.year
        test_df['Month'] = test_df['Date'].dt.month
        test_df['CompetitionOpen_missing'] = test_df['CompetitionOpenSinceYear'].isna().astype(int)
        test_df['CompetitionOpenSinceMonth'].fillna(1,inplace=True)
        test_df['CompetitionOpenSinceYear'].fillna(test_df['Year'].min(),inplace=True)
        test_df['CompetitionOpen'] = 12 * (test_df['Year'] - test_df['CompetitionOpenSinceYear']) + (test_df['Month'] - test_df['CompetitionOpenSinceMonth'])
        test_df['CompetitionOpen'] = test_df['CompetitionOpen'].apply(lambda x: max(x, 0))

        logging.info('Imputed CompetitionDistance, CompetitionOpenSinceMonth/Year')

        train_df['WeekOfYear'] = train_df['Date'].dt.isocalendar().week
        train_df['Promo2OpenSinceMonths'] = 12 * (train_df['Year'] - train_df['Promo2SinceYear']) + (train_df['WeekOfYear'] - train_df['Promo2SinceWeek']) / 4.0
        train_df['Promo2OpenSinceMonths'] = train_df['Promo2OpenSinceMonths'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)
        train_df.loc[train_df['Promo2'] == 0, 'Promo2OpenSinceMonths'] = 0
        month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        train_df['MonthStr'] = train_df['Month'].map(month_map)
        train_promo_months = train_df['PromoInterval'].fillna('').str.split(',')
        train_df['IsPromoMonth'] = [
            1 if m in months else 0 
            for m, months in zip(train_df['MonthStr'], train_promo_months)
        ]

        test_df['WeekOfYear'] = test_df['Date'].dt.isocalendar().week
        test_df['Promo2OpenSinceMonths'] = 12 * (test_df['Year'] - test_df['Promo2SinceYear']) + (test_df['WeekOfYear'] - test_df['Promo2SinceWeek']) / 4.0
        test_df['Promo2OpenSinceMonths'] = test_df['Promo2OpenSinceMonths'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)
        test_df.loc[test_df['Promo2'] == 0, 'Promo2OpenSinceMonths'] = 0
        test_df['MonthStr'] = test_df['Month'].map(month_map)
        test_promo_months = test_df['PromoInterval'].fillna('').str.split(',')
        test_df['IsPromoMonth'] = [
            1 if m in months else 0 
            for m, months in zip(train_df['MonthStr'], test_promo_months)
        ]

        logging.info('Imputed Promo2SinceWeek/Year and PromoInterval')

        return train_df,test_df

    except Exception as e:
        raise CustomException(e,sys)


def feature_engineering(train_df,test_df):
        try:
            train_df = train_df[train_df['Open']==1].copy()
            test_df = test_df[test_df['Open']==1].copy()

            train_df = train_df[train_df['Date'] != pd.Timestamp('2015-07-04')]
            test_df = test_df[test_df['Date'] != pd.Timestamp('2015-07-04')]

            train_df['StateHoliday'] = np.where((train_df['StateHoliday'] == '0') | (train_df['StateHoliday'] == 0),0,1)
            test_df['StateHoliday'] = np.where((test_df['StateHoliday'] == '0') | (test_df['StateHoliday'] == 0),0,1)

            train_df['Assortment'] = np.where(train_df['Assortment'] == 'b','b','other')
            test_df['Assortment'] = np.where(test_df['Assortment'] == 'b','b','other')

            store_avg_sales = train_df.groupby('Store')['Sales'].mean().rename('Store_avg_sales')
            store_avg_customers = train_df.groupby('Store')['Customers'].mean().rename('Store_avg_customers')
            train_df = train_df.merge(store_avg_sales, on='Store', how='left')
            train_df = train_df.merge(store_avg_customers, on='Store', how='left')
            test_df = test_df.merge(store_avg_sales, on='Store', how='left')
            test_df = test_df.merge(store_avg_customers, on='Store', how='left')

            drop_columns = ['Customers','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','MonthStr']

            for col in drop_columns:
                if col in train_df.columns:
                    train_df.drop(columns=[col],inplace=True)
                if col in test_df.columns:
                    test_df.drop(columns=[col],inplace=True)

            return train_df,test_df
        
        except Exception as e:
            raise CustomException(e,sys)

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def model_training(X_train,X_test,y_train,y_test):
    try:
        model = XGBRegressor(
            tree_method='hist', n_jobs=-1, random_state=42,
            max_depth=10, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8)
        model.fit(X_train,y_train)
        pred_log = model.predict(X_test)
        prediction = np.expm1(pred_log)

        def rmspe(y_true, y_pred):
                mask = y_true != 0
                return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))
        
        score = rmspe(y_test.values, prediction)

        return score*100,model

    except Exception as e:
        raise CustomException(e,sys)
