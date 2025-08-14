import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile

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
# st.write("Poverty Risk Prediction App based on UNSDG 1 indicators")
st.markdown("""
PovertyLens is a Poverty Risk Prediction App based on UNSDG 1 indicators.
Select a country to view its poverty risk, compare with global averages, and explore historical trends.
""")

country = st.selectbox("Select Country", df['Country'].unique())
row = df[df['Country'] == country]

if not row.empty:
    # map visualization
    st.subheader(f"Country Map: {country}")
    if "ISO" in row.columns:
        iso_code = row['ISO'].values[0]
        map_df = pd.DataFrame({"iso_alpha": [iso_code], "Selected": [1]})
        fig_map = px.choropleth(
            map_df,
            locations="iso_alpha",
            color="Selected",
            color_continuous_scale=["green", "green"],  # Always green
            locationmode="ISO-3",
            scope="world"
        )
        fig_map.update_coloraxes(showscale=False)  # Hide color bar
        st.plotly_chart(fig_map)
    else:
        st.info("Add an ISO column to your dataset for map visualization.")

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
    st.markdown("This section shows the predicted poverty risk for the selected country in 2025 based on socioeconomic indicators.")
    st.success(f"Poverty Risk (2025): {pred:.2%}")

    st.subheader("Key Poverty Indicators")
    st.info("Poverty Headcount Ratio at `$2.15` per day: Percentage of the population living on less than `$2.15` per day (2017 PPP). This is the international extreme poverty line.")
    st.write(f"- Poverty Headcount Ratio at $2.15/day (2025) in {country}: {row['2025_215'].values[0]:.2%}")
    st.info("Poverty Headcount Ratio at `$3.65` per day: Percentage of the population living on less than `$3.65` per day (2017 PPP). This is a higher poverty threshold used for lower-middle income countries.")
    st.write(f"- Poverty Headcount Ratio at $3.65/day (2025) in {country}: {row['2025_365'].values[0]:.2%}")

    # show after taxes and transfers only if available
    st.info("Poverty Rate After Taxes and Transfers: Share of people living below the poverty line after government taxes and social transfers. Shows the impact of social protection policies.")
    if '2025' in row.columns and not pd.isna(row['2025'].values[0]):
        val_post_tax = row['2025'].values[0]
        st.write(f"- Poverty Rate After Taxes and Transfers (2025) in {country}: {val_post_tax:.2%}")
    else:
        val_post_tax = None  
        st.write(f"- Poverty Rate After Taxes and Transfers (2025) in {country}: Data not available")

    # plot country vs global average graph
    st.subheader(f"{country} vs Global Average (2025)")
    st.markdown(
        f"This chart compares the {country} poverty indicators for 2025 with the global average. "
        "It helps you see how the country stands relative to the rest of the world for each indicator."
    )

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

    #description
    desc_lines = []
    for i, label in enumerate(compare_df['Indicator']):
        country_val = compare_df['Selected Country'][i]
        global_val = compare_df['Global Average'][i]
        if country_val is not None and global_val is not None:
            if country_val > global_val:
                desc_lines.append(
                    f"The average {label.lower()} in {country} (**{country_val:.2%}**) is **above** the global average (**{global_val:.2%}**)."
                )
            else:
                desc_lines.append(
                    f"The average {label.lower()} in {country} (**{country_val:.2%}**) is **below** the global average (**{global_val:.2%}**)."
                )
        else:
            desc_lines.append(
                f"Data for {label.lower()} is not available for {country} or global average."
            )
    st.markdown("**Insights:**\n" + "\n".join([f"- {line}" for line in desc_lines]))

    # plot the historical indicators trend
    st.subheader(f"{country} Poverty Indicator Trends")
    st.markdown(
        f"This chart shows the historical trends of poverty indicators in {country} from 2000 to 2025. "
        "You can observe how the poverty headcount ratios have changed over time and compare their patterns."
    )

    trend_indicators = {
        "Poverty Headcount Ratio at $2.15/day": [f"{year}_215" for year in range(2000, 2026)],
        "Poverty Headcount Ratio at $3.65/day": [f"{year}_365" for year in range(2000, 2026)]
    }

    trend_data = pd.DataFrame({"Year": list(range(2000, 2026))})
    for label, cols in trend_indicators.items():
        values = [row[col].values[0] if col in row.columns and not pd.isna(row[col].values[0]) else None for col in cols]
        trend_data[label] = values

    trend_melted = trend_data.melt(id_vars="Year", var_name="Indicator", value_name="Value")

    fig_trend = px.line(
        trend_melted,
        x="Year",
        y="Value",
        color="Indicator",
        template="plotly_dark",
        markers=True
    )
    fig_trend.update_layout(
        xaxis_title="Year",
        yaxis_title="Value",
        legend_title_text="Indicator"
    )
    st.plotly_chart(fig_trend)

    #description
    insights = []
    for label in trend_indicators.keys():
        series = trend_data[label].dropna()
        if not series.empty:
            start_val = series.iloc[0]
            end_val = series.iloc[-1]
            if end_val > start_val:
                insights.append(
                    f"- The {label.lower()} in {country} increased from {start_val:.2%} in 2000 to {end_val:.2%} in 2025."
                )
            elif end_val < start_val:
                insights.append(
                    f"- The {label.lower()} in {country} decreased from {start_val:.2%} in 2000 to {end_val:.2%} in 2025."
                )
            else:
                insights.append(
                    f"- The {label.lower()} in {country} remained stable at {start_val:.2%} from 2000 to 2025."
                )
        else:
            insights.append(f"- Data for {label.lower()} in {country} is not available for trend analysis.")
    st.markdown("**Trend Insights:**\n" + "\n".join(insights))

    # smart advice
    st.subheader("Smart Advice")   
    
    advice_lines=[]
    # advice for $2.15/day vs global average indicator
    if row['2025_215'].values[0] > df['2025_215'].mean():
        advice_lines.append(
            f"- {country} has a higher poverty headcount ratio at $2.15/day than the global average. Consider strengthening social safety nets and targeted poverty alleviation programs."
        )
    else:
        advice_lines.append(
            f"- {country}'s extreme poverty rate at $2.15/day is below the global average. Maintain current policies and monitor for emerging risks."
        )

    if '2025' in row.columns and not pd.isna(row['2025'].values[0]):
        if row['2025'].values[0] > df['2025'].mean():
            advice_lines.append(
                "- The poverty rate after taxes and transfers is above the global average. Review and enhance social protection and tax policies."
            )
        else:
            advice_lines.append(
                "- Social protection policies are effective compared to global average. Continue to invest in inclusive programs."
            )

    # Advice for $3.65/day vs global advice indicator
    if row['2025_365'].values[0] > df['2025_365'].mean():
        advice_lines.append(
            f"- {country} has a higher poverty headcount ratio at $3.65/day than the global average. Focus on inclusive economic growth and job creation."
        )
    else:
        advice_lines.append(
            f"- {country}'s extreme poverty rate at $3.65/day is below the global average. Maintain efforts to support vulnerable groups."
        )

    # advice based on trend ($2.15/day)
    series_215 = trend_data["Poverty Headcount Ratio at $2.15/day"].dropna()
    if not series_215.empty and series_215.iloc[-1] > series_215.iloc[0]:
        advice_lines.append(
            "- The poverty headcount ratio at $2.15/day has increased over time. Investigate causes and strengthen poverty reduction strategies."
        )
    elif not series_215.empty and series_215.iloc[-1] < series_215.iloc[0]:
        advice_lines.append(
            "- The poverty headcount ratio at $2.15/day has decreased over time. Continue successful interventions and monitor progress."
        )

    # advice based on trend ($3.65/day)
    series_365 = trend_data["Poverty Headcount Ratio at $3.65/day"].dropna()
    if not series_365.empty and series_365.iloc[-1] > series_365.iloc[0]:
        advice_lines.append(
            "- The poverty headcount ratio at $3.65/day has increased over time. Investigate causes and strengthen poverty reduction strategies."
        )
    elif not series_365.empty and series_365.iloc[-1] < series_365.iloc[0]:
        advice_lines.append(
            "- The poverty headcount ratio at $3.65/day has decreased over time. Continue successful interventions and monitor progress."
        )

    # general advice
    advice_lines.append(
        "- Consider investing in education, healthcare, and social protection to further reduce poverty and improve well-being."
    )
    st.markdown("\n".join(advice_lines))

    # report data
    st.subheader("Downloadable Report")

    def create_pdf(country, pred, row, val_post_tax, desc_lines, insights, advice_lines, fig, fig_trend):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        # title
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(0, 70, 140)
        pdf.cell(0, 15, f"PovertyLens Report: {country}", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        # section separator
        pdf.set_draw_color(0, 70, 140)
        pdf.set_line_width(1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        # prediction
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Prediction Result", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 10, f"Poverty Risk (2025): {pred:.2%}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        # key indicators
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Key Poverty Indicators", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, f"- Poverty Headcount Ratio at $2.15/day (2025): {row['2025_215'].values[0]:.2%}", ln=True)
        pdf.cell(0, 8, f"- Poverty Headcount Ratio at $3.65/day (2025): {row['2025_365'].values[0]:.2%}", ln=True)
        pdf.cell(0, 8, f"- Poverty Rate After Taxes and Transfers (2025): {val_post_tax if val_post_tax else 'Data not available'}", ln=True)
        pdf.ln(5)
        # section separator
        pdf.set_draw_color(220, 220, 220)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        # insights
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Insights", ln=True)
        pdf.set_font("Arial", '', 12)
        for line in desc_lines:
            pdf.multi_cell(0, 8, f"- {line}")
        pdf.ln(3)
        # trend insights
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Trend Insights", ln=True)
        pdf.set_font("Arial", '', 12)
        for line in insights:
            pdf.multi_cell(0, 8, line)
        pdf.ln(3)
        # smart a dvice
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Smart Advice", ln=True)
        pdf.set_font("Arial", '', 12)
        for line in advice_lines:
            pdf.multi_cell(0, 8, line)
        pdf.ln(5)
        # section separator
        pdf.set_draw_color(0, 70, 140)
        pdf.set_line_width(1)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        # save charts as images and add to PDF
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile1:
            fig.write_image(tmpfile1.name)
            pdf.image(tmpfile1.name, w=180)
        pdf.ln(5)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile2:
            fig_trend.write_image(tmpfile2.name)
            pdf.image(tmpfile2.name, w=180)
        pdf.ln(5)
        # footer
        pdf.set_y(-20)
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 10, "Generated by PovertyLens | github.com/priscillahandoyo/poverty-lens", 0, 0, 'C')
        # save PDF to bytes
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
            pdf.output(pdf_file.name)
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
        return pdf_bytes

    # button to download PDF
    if st.button("Download Full Report as PDF"):
        val_post_tax_str = f"{val_post_tax:.2%}" if val_post_tax is not None else "Data not available"
        pdf_bytes = create_pdf(
            country, pred, row, val_post_tax_str, desc_lines, insights, advice_lines, fig, fig_trend
        )
        st.download_button(
            label="Click to Download PDF",
            data=pdf_bytes,
            file_name=f"{country}_poverty_report.pdf",
            mime="application/pdf"
        )

else:
    st.warning("Country data not found.")
