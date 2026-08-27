import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

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

        train_df.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','MonthStr'], axis=1, inplace=True)
        test_df.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','MonthStr'], axis=1, inplace=True)

        return train_df,test_df

    except Exception as e:
        raise CustomException(e,sys)