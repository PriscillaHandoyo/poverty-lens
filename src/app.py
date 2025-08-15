import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile

# set up wide streamlit layout
st.set_page_config(layout="wide")

DATASET_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data/dataset')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/poverty_risk_model.pkl')

# load model and data
model = joblib.load(MODEL_PATH)
df_215 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-2-15-day.csv'))
df_365 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-3-65-day.csv'))
df_post_tax = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-rate-after-taxes-and-transfers.csv'))
# df_unemployment = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-unemployment-rate.csv'))

# merge data
df = df_215.merge(df_365, on=['ISO', 'Country'], suffixes=('_215', '_365'))
df = df.merge(df_post_tax, on=['ISO', 'Country'], suffixes=('', '_post_tax'))
# df = df.merge(df_unemployment, on='Country', how='left')

# sidebar layout for country selection and simulation
st.sidebar.title("Country & Policy Simulation")

country = st.sidebar.selectbox("Select Country", df['Country'].unique())
row = df[df['Country'] == country]

# year selection
available_years = sorted([int(col.split('_')[0]) for col in df.columns if '_215' in col])
year = st.sidebar.selectbox("Select Year", available_years, index=available_years.index(2025) if 2025 in available_years else 0)

# model feature columns
model_features = model.feature_names_in_

# default values from data (if available)
col_215 = f"{year}_215"
col_365 = f"{year}_365"
default_215 = row['2025_215'].values[0] if not row.empty and '2025_215' in row.columns else 0.0
default_365 = row['2025_365'].values[0] if not row.empty and '2025_365' in row.columns else 0.0

st.sidebar.markdown("#### Simulate Policy Changes")
sim_215 = st.sidebar.slider(
    f"Poverty Headcount Ratio at $2.15/day ({year})", min_value=0.0, max_value=1.0, value=float(default_215), step=0.01,
    help="Adjust to simulate changes in extreme poverty rate."
)
sim_365 = st.sidebar.slider(
    f"Poverty Headcount Ratio at $3.65/day ({year})", min_value=0.0, max_value=1.0, value=float(default_365), step=0.01,
    help="Adjust to simulate changes in moderate poverty rate."
)

# literacy rate slider
# col_unemployment = f"{year}_unemployment"
# default_unemployment = row[col_unemployment].values[0] if not row.empty and col_unemployment in row.columns else 0.0

# sim_unemployment = st.sidebar.slider(
#     f"Unemployment Rate (%) ({year})", min_value=0.0, max_value=100.0, value=float(default_unemployment), step=0.1,
#     help="Adjust to simulate changes in unemployment rate."
# )

# print(df[[ 'Country', f'{year}_literacy' ]].head(10))

simulate = st.sidebar.button("Simulate Policy Changes")

# st.write("Columns:", df.columns.tolist())

if not row.empty:
    # first row: title, description, and logo
    first_row = st.columns([6, 3, 1])
    with first_row[0]:
        st.title("PovertyLens")
        st.markdown("""
        PovertyLens is a Poverty Risk Prediction App based on UNSDG 1 indicators.
        Select a country to view its poverty risk, compare with global averages, and explore historical trends.
        """)
    with first_row[1]:
        st.write("")  # Spacer for alignment
    with first_row[2]:
        st.image(os.path.join(os.path.dirname(__file__), "../readme_data/logo.png"), width=80)

    # second row: map visualization and prediction result
    second_row = st.columns([7, 5])
    with second_row[0]:
        st.header(f"Country Map: {country}")
        if "ISO" in row.columns:
            iso_code = row['ISO'].values[0]
            map_df = pd.DataFrame({"iso_alpha": [iso_code], "Selected": [1]})
            fig_map = px.choropleth(
                map_df,
                locations="iso_alpha",
                color="Selected",
                color_continuous_scale=["#4cbb17", "#4cbb17"],
                locationmode="ISO-3",
                scope="world"
            )
            fig_map.update_coloraxes(showscale=False)
            st.plotly_chart(fig_map, use_container_width=True, key="country_map")
        else:
            st.info("Add an ISO column to your dataset for map visualization.")

    with second_row[1]:
        # use only the columns the model expects
        X = row[model_features].copy()
        # Replace selected year columns with simulated values
        if col_215 in X.columns:
            X.at[X.index[0], col_215] = sim_215
        if col_365 in X.columns:
            X.at[X.index[0], col_365] = sim_365
        # if col_unemployment in X.columns:
        #     X.at[X.index[0], col_unemployment] = sim_unemployment

        pred = model.predict(X)[0]
        st.header("Prediction Result")
        st.caption(f"This section shows the predicted poverty risk for {country} in 2025 based on socioeconomic indicators.")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 280px;">
                <span style="font-size: 8rem; font-weight: bold; color: #4cbb17;">{pred:.2%}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # literacy rate
    # if col_literacy in X.columns:
    #     X.at[X.index[0], col_literacy] = sim_literacy
    
    # st.subheader("Literacy Rate")
    # st.info("Literacy Rate: Percentage of people aged 15 and above who can read and write. Higher literacy is linked to lower poverty risk.")
    # st.write(f"- Literacy Rate ({year}) in {country}: {sim_literacy:.1f}% (simulated)")

    # third row: key indicators
    st.header("Key Poverty Indicators")
    val_215 = sim_215
    val_365 = sim_365
    # after taxes and transfers
    col_post_tax = f"{year}"
    if col_post_tax in row.columns and not pd.isna(row[col_post_tax].values[0]):
        val_post_tax = row[col_post_tax].values[0]
    else:
        val_post_tax = None
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=f"Poverty Headcount Ratio at $2.15/day ({year})",
            value=f"{val_215:.2%}" if val_215 is not None else "N/A",
            help="Poverty Headcount Ratio at `$2.15` per day: Percentage of the population living on less than `$2.15` per day (2017 PPP). This is the international extreme poverty line."
        )

    with col2:
        st.metric(
            label=f"Poverty Headcount Ratio at $3.65/day ({year})",
            value=f"{val_365:.2%}" if val_365 is not None else "N/A",
            help="Poverty Headcount Ratio at `$3.65` per day: Percentage of the population living on less than `$3.65` per day (2017 PPP). This is a higher poverty threshold used for lower-middle income countries."
        )

    with col3:
        st.metric(
            label=f"Poverty Rate After Taxes and Transfers ({year})",
            value=f"{val_post_tax:.2%}" if val_post_tax is not None else "N/A",
            help="Poverty Rate After Taxes and Transfers: Share of people living below the poverty line after government taxes and social transfers. Shows the impact of social protection policies."
        )
    
    # add some space between rows (indicators)
    # st.markdown("<br>", unsafe_allow_html=True)  

    # col_unemployment = st.columns(1)
    # with col_unemployment[0]:
    #     st.metric(
    #         label=f"Unemployment Rate ({year})",
    #         value=f"{sim_unemployment:.1f}%" if sim_unemployment is not None else "N/A",
    #         help="Unemployment Rate: Percentage of the labor force that is jobless. Higher unemployment can increase poverty risk."
    #     )
    
    # add some space
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("<br>", unsafe_allow_html=True)

    # plot country vs global average graph
    indicators = {
        "Poverty Headcount Ratio at $2.15/day": col_215,
        "Poverty Headcount Ratio at $3.65/day": col_365,
        "Poverty Rate After Taxes and Transfers": col_post_tax
    }

    selected_country_values = [val_215, val_365, val_post_tax if val_post_tax is not None else None]
    global_avg_values = [df[col_215].mean() if col_215 in df.columns else None,
                        df[col_365].mean() if col_365 in df.columns else None,
                        df[col_post_tax].mean() if col_post_tax in df.columns else None]

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
        legend_title_text="Variable",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.3,
            xanchor="center",
            x=0.5,
        )
    )

    #plot the country vs global average chart
    st.header(f"**{country} vs Global Average ({year})**")
    st.caption(
            f"This chart compares the {country} poverty indicators for {year} with the global average. "
            "It helps you see how the country stands relative to the rest of the world for each indicator."
    )
    st.plotly_chart(fig, use_container_width=True, key="country_vs_global")
    # insight
    with st.expander("Show Insights"):
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
        st.markdown("\n".join([f"- {line}" for line in desc_lines]))

    # add some space
    st.markdown("<br>", unsafe_allow_html=True) 

    # plot the historical indicators trend
    trend_indicators = {
        "Poverty Headcount Ratio at $2.15/day": [f"{year}_215" for year in range(2000, 2026)],
        "Poverty Headcount Ratio at $3.65/day": [f"{year}_365" for year in range(2000, 2026)],
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
        markers=True,
        height=400
    )
    fig_trend.update_layout(
        xaxis_title="Year",
        yaxis_title="Value",
        legend_title_text="Indicator",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.3,
            xanchor="center",
            x=0.5
        )
    )

    # plot the historical indicators trend on poverty ratio after taxes and transfers
    trend_tt_indicators = {
        "Poverty Rate After Taxes and Transfers": [str(year) for year in range(2000, 2026)]
    }

    trend_tt_data = pd.DataFrame({"Year": list(range(2000, 2026))})
    for label, cols in trend_tt_indicators.items():
        values = [row[col].values[0] if col in row.columns and not pd.isna(row[col].values[0]) else None for col in cols]
        trend_tt_data[label] = values

    trend_tt_melted = trend_tt_data.melt(id_vars="Year", var_name="Indicator", value_name="Value")

    fig_tt_trend = px.line(
        trend_tt_melted,
        x="Year",
        y="Value",
        color="Indicator",
        template="plotly_dark",
        markers=True,
        height=400
    )
    fig_tt_trend.update_layout(
        xaxis_title="Year",
        yaxis_title="Value",
        legend_title_text="Indicator",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.3,
            xanchor="center",
            x=0.5
        )
    )

    # fourth row: country vs global average and trend charts
    fourth_row = st.columns([6, 6])
    with fourth_row[0]:
        st.header(f"{country} Poverty Rate After Taxes and Transfers Trend")
        st.caption(
            f"This chart shows the historical trend of the poverty rate after taxes and transfers in {country} from 2000 to 2025. "
            "It helps you understand how social protection policies and government interventions have impacted poverty over time."
        )
        st.plotly_chart(fig_tt_trend, use_container_width=True, key="tt_trend")
        # insight
        with st.expander("Show Insights"):
            series_tt = trend_tt_data["Poverty Rate After Taxes and Transfers"].dropna()
            # filter 0 and NaN values for average calculation
            nonzero_series = series_tt[series_tt != 0]
            if not series_tt.empty:
                # highest value and year
                max_val = series_tt.max()
                max_idx = series_tt.idxmax()
                max_year = trend_tt_data.loc[max_idx, "Year"]
                # lowest value and year
                min_val = series_tt.min()
                min_idx = series_tt.idxmin()
                min_year = trend_tt_data.loc[min_idx, "Year"]
                # average
                avg_val = nonzero_series.mean() if not nonzero_series.empty else 0
                st.markdown(
                   f"- The **highest poverty rate after taxes and transfers** was **{max_val:.2%}** in **{max_year}**."
                )
                st.markdown(
                    f"- The **lowest poverty rate after taxes and transfers** was **{min_val:.2%}** in **{min_year}**."
                )
                if not nonzero_series.empty:
                    st.markdown(
                        f"- The **average poverty rate after taxes and transfers** (excluding 0 and missing data) is **{avg_val:.2%}**."
                    )
                else:
                    st.markdown(
                        "- There is no non-zero data available to calculate the average poverty rate after taxes and transfers."
                    )
            else:
                st.markdown("- Data for poverty rate after taxes and transfers is not available for trend analysis.")

    with fourth_row[1]:
        st.header(f"**{country} Poverty Indicator Trends**")
        st.caption(
            f"This chart shows the historical trends of poverty indicators in {country} from 2000 to 2025. "
            "You can observe how the poverty headcount ratios have changed over time and compare their patterns."
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="indicator_trends")
        # insights
        with st.expander("Show Insights"):
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
            st.markdown("\n".join(insights))

    # smart advice
    st.header("Smart Advice")   
    advice_lines=[]
    # advice for $2.15/day vs global average indicator
    if val_215 is not None and global_avg_values[0] is not None:
        if val_215 > global_avg_values[0]:
            advice_lines.append(
                f"- {country} has a higher poverty headcount ratio at $2.15/day than the global average. Consider strengthening social safety nets and targeted poverty alleviation programs."
            )
        else:
            advice_lines.append(
                f"- {country}'s extreme poverty rate at $2.15/day is below the global average. Maintain current policies and monitor for emerging risks."
            )

    if val_post_tax is not None and col_post_tax in df.columns and df[col_post_tax].mean() is not None:
        if val_post_tax > df[col_post_tax].mean():
            advice_lines.append(
                "- The poverty rate after taxes and transfers is above the global average. Review and enhance social protection and tax policies."
            )
        else:
            advice_lines.append(
                "- Social protection policies are effective compared to global average. Continue to invest in inclusive programs."
            )

    # advice for $3.65/day vs global advice indicator
    if val_365 is not None and global_avg_values[1] is not None:
        if val_365 > global_avg_values[1]:
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
            f"- The poverty headcount ratio at $2.15/day has increased over time. Investigate causes and strengthen poverty reduction strategies."
        )
    elif not series_215.empty and series_215.iloc[-1] < series_215.iloc[0]:
        advice_lines.append(
            f"- The poverty headcount ratio at $2.15/day has decreased over time. Continue successful interventions and monitor progress."
        )

    # advice based on trend ($3.65/day)
    series_365 = trend_data["Poverty Headcount Ratio at $3.65/day"].dropna()
    if not series_365.empty and series_365.iloc[-1] > series_365.iloc[0]:
        advice_lines.append(
            f"- The poverty headcount ratio at $3.65/day has increased over time. Investigate causes and strengthen poverty reduction strategies."
        )
    elif not series_365.empty and series_365.iloc[-1] < series_365.iloc[0]:
        advice_lines.append(
            f"- The poverty headcount ratio at $3.65/day has decreased over time. Continue successful interventions and monitor progress."
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
