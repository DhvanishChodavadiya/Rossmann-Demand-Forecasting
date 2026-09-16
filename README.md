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