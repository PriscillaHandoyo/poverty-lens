import os
import streamlit as st
import pandas as pd
import joblib

DATASET_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data/dataset')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/poverty_risk_model.pkl')

# load model and data
model = joblib.load(MODEL_PATH)
df_215 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-2-15-day.csv'))
df_365 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-3-65-day.csv'))
df_post_tax = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-rate-after-taxes-and-transfers.csv'))

# merge data
df = df_215.merge(df_365, on=['ISO', 'Country'], suffixes=('_215', '_365'))
df = df.merge(df_post_tax, on=['ISO', 'Country'], suffixes=('', '_post_tax'))

st.write("Columns:", df.columns.tolist())

st.title("PovertyLens")
st.write("Poverty Risk Prediction App based on UNSDG 1 indicators")

country = st.selectbox("Select Country", df['Country'].unique())
row = df[df['Country'] == country]

if not row.empty:
    # only use _215 and _365 features for prediction
    feature_cols = []
    for suffix in ['_215', '_365']:
        for year in range(2000, 2025):
            col = f"{year}{suffix}"
            if col in df.columns:
                feature_cols.append(col)
    X = row[feature_cols]
    pred = model.predict(X)[0]
    st.subheader("Prediction Result")
    st.success(f"Poverty Risk (2025): {pred:.2%}")

    st.subheader("Smart Insights")
    st.write(f"- Poverty Headcount Ratio at $2.15/day (2025): {row['2025_215'].values[0]:.2%}")
    st.write(f"- Poverty Headcount Ratio at $3.65/day (2025): {row['2025_365'].values[0]:.2%}")

    # show after taxes and transfers only if available
    if '2025' in row.columns and not pd.isna(row['2025'].values[0]):
        val_post_tax = row['2025'].values[0]
        st.write(f"- Poverty Rate After Taxes and Transfers (2025): {val_post_tax:.2%}")
    else:
        st.write("- Poverty Rate After Taxes and Transfers (2025): Data not available")
else:
    st.warning("Country data not found.")