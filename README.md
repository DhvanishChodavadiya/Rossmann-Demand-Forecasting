Python version = 3.14.7

# Dependencies
To install dependencies run this command in terminal:
pip install -r requirements.txt

# Data split
- Data is splitted according to "Date" feature
- train set : All records before "2015-06-01"
- valid set : All records after "2015-06-01"

# Missing value imputation
- "CompetitionDistance" : 0.26% 
    -> imputed with 0

- "CompetitionOpenSinceMonth/Year" : 31.78% 
    -> imputed with minimum value in "Month" and "Year" feature(Both feature are extracted from "Date" feature).
    -> Add "CompetitionOpen" feature(Shows since how many months competition was opened)
    -> Add "CompetitionOpen_missing" flag.
    -> Drop raw features("CompetitionOpenSinceMonth/Year").

- "Promo2SinceWeek/Year","PromoInterval" : 49.94%
    -> Values are missing, where store does not participating in Promo2.
    -> "Promo2" working as a missing flag.
    -> Add "Promo2OpenSinceMonths" feature.
    -> "PromoInterval" imputed with empty space.
    -> Add "IsPromoMonth" feature.
    -> Drop raw features("Promo2SinceWeek/Year","PromoInterval").

# EDA
- Consider only records with "Open" == 1.

# Feature Engineering
- log1p transform to y_train(rmspe = 21.45%). There are some outliers in "Sales", So log1p transforms data into specific range.
- Create new features "Store_avg_sales" and "Store_avg_customers" on train set, and then merge with train and valid set(rmspe = 18.04). Create both feature on "Store" feature.
- Turned "Assortment" feature values into 2 categories : 'b' and 'other'(rmspe : 16.27%)
- There was a noise in data. Worst 30 rows where prediction were most worse, 7 out of them rows were with 'Date' == '2015-07-04'. After a little research, it was founded that there was extreme heatwave across the Germany and a Bavarian town 'Kitzingen' recorded highest temperature of history of Germany on that day. And also there was female FIFA world cup match between Germany and England. So that noise was removed from data(15.63%).
- After Hyperparameter Tuning(13.89%).
- After that i checked that either model is overfittied or underfitted. So, I also calculated rmspe on training data and it's confirmed that model is normal(training rmspe = 13.56%)


# Rossmann Sales Predictor — Streamlit App

## Files
- `app.py` — the Streamlit app
- `model.pkl` — trained XGBoost model
- `preprocessor.pkl` — fitted ColumnTransformer (StandardScaler + OneHotEncoder)
- `store_lookup.csv` — one row per store with static/engineered features
- `requirements.txt` — dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. User picks a Store ID from the sidebar — the app looks up that store's
   static features (StoreType, Assortment, CompetitionDistance, historical
   average sales/customers, Promo2 status) from `store_lookup.csv`.
2. User fills in day-specific details (date, Open, Promo, holidays).
3. The app builds a single-row DataFrame matching the exact columns the
   `preprocessor` was fit on, transforms it, and calls `model.predict()`.
4. Since the model was trained on log1p(Sales), the raw prediction is
   inverse-transformed with `np.expm1()` before being shown.

## Known limitations
- The model was trained only on 2013-2015 data (Year is one-hot encoded:
  Year_2014/Year_2015), so predictions for dates outside that window are
  extrapolations and won't be reliable.
- `IsPromoMonth` is a manual toggle since `store_lookup.csv` only stores the
  already-computed `Promo2OpenSinceMonths`, not the raw `PromoInterval`
  string. If you want this auto-derived, add `PromoInterval` to
  `store_lookup.csv` and compute `IsPromoMonth` from the selected date.