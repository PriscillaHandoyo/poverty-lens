import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

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

# st.write("Columns:", df.columns.tolist())

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

    # plot country vs global average graph
    st.subheader(f"{country} vs Global Average (2025)")

    indicators = {
        "Poverty Headcount Ratio at $2.15/day": "2025_215",
        "Poverty Headcount Ratio at $3.65/day": "2025_365"
    }

    selected_country_values = []
    global_avg_values = []
    for label, col in indicators.items():
        country_val = row[col].values[0] if col in row.columns and not pd.isna(row[col].values[0]) else None
        global_val = df[col].mean() if col in df.columns else None
        selected_country_values.append(country_val)
        global_avg_values.append(global_val)

    compare_df = pd.DataFrame({
        'Indicator': list(indicators.keys()),
        'Selected Country': selected_country_values,
        'Global Average': global_avg_values
    })

    fig = px.bar(
        compare_df,
        x='Indicator',
        y=['Selected Country', 'Global Average'],
        barmode='group',
        color_discrete_map={"Selected Country": "deepskyblue", "Global Average": "orange"},
        template="plotly_dark"
    )
    fig.update_layout(
        xaxis_title="Indicator",
        yaxis_title="Value",
        legend_title_text="Variable"
    )
    st.plotly_chart(fig)

else:
    st.warning("Country data not found.")
