import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from pathlib import Path

# Resolve paths relative to this script's own folder, so the app works
# no matter what directory it's launched from.
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "artifacts/model.pkl"
PREPROCESSOR_PATH = APP_DIR / "artifacts/preprocessor.pkl"
STORE_LOOKUP_PATH = APP_DIR / "store_lookup.csv"

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Rossmann Sales Predictor", page_icon="🛒", layout="centered")

st.title("🛒 Rossmann Store Sales Predictor")
st.write(
    "Predict a store's daily sales using an XGBoost model trained on the "
    "Rossmann Store Sales dataset. Pick a store and a few day-specific details below."
)

# ----------------------------
# Load model, preprocessor, store lookup (cached so this only runs once)
# ----------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.error(
            f"Could not find model.pkl / preprocessor.pkl next to app.py "
            f"(looked in: {APP_DIR}). Make sure all files are in the same folder."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

@st.cache_data
def load_store_lookup():
    if not STORE_LOOKUP_PATH.exists():
        st.error(
            f"Could not find store_lookup.csv next to app.py (looked in: {APP_DIR})."
        )
        st.stop()
    return pd.read_csv(STORE_LOOKUP_PATH)

model, preprocessor = load_artifacts()
store_df = load_store_lookup()

# ----------------------------
# Sidebar: Store selection (auto-fills static features)
# ----------------------------
st.sidebar.header("1. Select a Store")

store_ids = sorted(store_df["Store"].unique().tolist())
selected_store = st.sidebar.selectbox("Store ID", store_ids)

store_row = store_df[store_df["Store"] == selected_store].iloc[0]

st.sidebar.markdown("**Store profile (auto-filled):**")
st.sidebar.write(f"- Store Type: `{store_row['StoreType']}`")
st.sidebar.write(f"- Assortment: `{store_row['Assortment']}`")
st.sidebar.write(f"- Competition Distance: `{store_row['CompetitionDistance']:.0f}` m")
st.sidebar.write(f"- Historical Avg Sales: `{store_row['Store_avg_sales']:.0f}`")
st.sidebar.write(f"- Historical Avg Customers: `{store_row['Store_avg_customers']:.0f}`")
st.sidebar.write(f"- Participates in Promo2: `{'Yes' if store_row['Promo2'] == 1 else 'No'}`")

# ----------------------------
# Main form: day-specific inputs
# ----------------------------
st.header("2. Day Details")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Date",
        value=date(2015, 7, 15),
        min_value=date(2013, 1, 1),
        max_value=date(2015, 12, 31),
        help="Model was trained on 2013–July 2015 data only. Predictions outside this "
         "range (or far from it) are not reliable.",
    )
    if selected_date > date(2015, 7, 31):
        st.warning(
            "⚠️ No ground-truth data exists after July 31, 2015 in this dataset, so "
            "predictions beyond this date are unvalidated extrapolations. The model's "
            "known accuracy (~13.89% RMSPE) only applies to Jun 2 – Jul 31, 2015."
        )
    is_open = st.selectbox("Is the store open on this day?", ["Yes", "No"], index=0)
    promo = st.selectbox("Running a Promo (daily discount) today?", ["No", "Yes"], index=0)

with col2:
    state_holiday = st.selectbox("Is this a State Holiday?", ["No", "Yes"], index=0)
    school_holiday = st.selectbox("Is this a School Holiday?", ["No", "Yes"], index=0)
    if store_row["Promo2"] == 1:
        is_promo_month = st.selectbox(
            "Is this a recurring Promo2 month for this store?",
            ["No", "Yes"],
            index=0,
            help="Promo2 runs only in specific recurring months per store "
                 "(e.g. Jan/Apr/Jul/Oct). Check your store's promo calendar.",
        )
    else:
        is_promo_month = "No"
        st.selectbox(
            "Is this a recurring Promo2 month for this store?",
            ["No (store does not participate in Promo2)"],
            index=0,
            disabled=True,
        )

# ----------------------------
# Build the feature row expected by the preprocessor
# ----------------------------
def build_feature_row(store_row, selected_date, is_open, promo, state_holiday, school_holiday, is_promo_month):
    year = selected_date.year
    month = selected_date.month
    day_of_week = selected_date.isoweekday()  # 1=Monday ... 7=Sunday, matches Rossmann's convention
    week_of_year = selected_date.isocalendar().week

    row = pd.DataFrame([{
        "CompetitionDistance": store_row["CompetitionDistance"],
        "CompetitionOpen": store_row["CompetitionOpen"],
        "Promo2OpenSinceMonths": store_row["Promo2OpenSinceMonths"],
        "Store_avg_sales": store_row["Store_avg_sales"],
        "Store_avg_customers": store_row["Store_avg_customers"],
        "Year": year,
        "StoreType": store_row["StoreType"],
        "Assortment": store_row["Assortment"],
        "Store": store_row["Store"],
        "DayOfWeek": day_of_week,
        "Open": 1 if is_open == "Yes" else 0,
        "Promo": 1 if promo == "Yes" else 0,
        "StateHoliday": 1 if state_holiday == "Yes" else 0,
        "SchoolHoliday": 1 if school_holiday == "Yes" else 0,
        "Promo2": store_row["Promo2"],
        "Month": month,
        "CompetitionOpen_missing": store_row["CompetitionOpen_missing"],
        "WeekOfYear": week_of_year,
        "IsPromoMonth": 1 if is_promo_month == "Yes" else 0,
    }])
    return row

# ----------------------------
# Predict
# ----------------------------
st.header("3. Prediction")

if st.button("Predict Sales", type="primary"):
    if is_open == "No":
        st.info("This store is marked as closed on this day, so predicted sales is **0**.")
    else:
        feature_row = build_feature_row(
            store_row, selected_date, is_open, promo, state_holiday, school_holiday, is_promo_month
        )
        X = preprocessor.transform(feature_row)
        pred_log = model.predict(X)
        prediction = np.expm1(pred_log)[0]

        st.metric(label=f"Predicted Sales — Store {selected_store} on {selected_date}", value=f"{prediction:,.0f}")

        with st.expander("See the exact inputs sent to the model"):
            st.table(feature_row.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Model: XGBoost, trained on log-transformed Sales with feature engineering "
    "(store-level historical averages, competition/promo duration, simplified "
    "Assortment) and time-based validation. Validation RMSPE ≈ 13.89%."
)
