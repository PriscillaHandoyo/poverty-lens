import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile

try:
    import kaleido
except ImportError:
    st.error("Kaleido is required for PDF export. Please install it with 'pip install kaleido'.")

# set up wide streamlit layout
st.set_page_config(layout="wide")

DATASET_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data/dataset')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/poverty_risk_model.pkl')

# load model and data
model = joblib.load(MODEL_PATH)
df_215 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-2-15-day.csv'))
df_365 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-3-65-day.csv'))
df_post_tax = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-rate-after-taxes-and-transfers.csv'))
df_unemployment = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-unemployment-rate.csv'))
df_literacy = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-literacy-rate.csv'))
df_employment = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-employment-to-population-ratio.csv'))

# merge data
df = df_215.merge(df_365, on=['ISO', 'Country'], suffixes=('_215', '_365'))
df = df.merge(df_post_tax, on=['ISO', 'Country'], suffixes=('', '_post_tax'))
df = df.merge(df_unemployment, on=['ISO', 'Country'], suffixes=('', '_unemployment'), how='left')
df = df.merge(df_literacy, on=['ISO', 'Country'], suffixes=('', '_literacy'), how='left')
df = df.merge(df_employment, on=['ISO', 'Country'], suffixes=('', '_employment'), how='left')

# sidebar layout for country selection and simulation
st.sidebar.title("Country & Policy Simulation")

country = st.sidebar.selectbox("Select Country", df['Country'].unique())
row = df[df['Country'] == country]

# year selection
available_years = sorted([int(col.split('_')[0]) for col in df.columns if '_215' in col])
year = st.sidebar.selectbox("Select Year", available_years, index=available_years.index(2025) if 2025 in available_years else 0)

# literacy rate for selected country and year
col_literacy = str(year)
if col_literacy in df_literacy.columns:
    literacy_row = df_literacy[df_literacy['Country'] == country]
    if not literacy_row.empty and not pd.isna(literacy_row[col_literacy].values[0]):
        val_literacy = literacy_row[col_literacy].values[0]
    else:
        val_literacy = None
else:
    val_literacy = None

# model feature columns
model_features = model.feature_names_in_

# default values from data (if available)
col_215 = f"{year}_215"
col_365 = f"{year}_365"  
# Get value from the original post-tax dataframe
col_unemployment = str(year)
if col_unemployment in df_unemployment.columns:
    unemployment_row = df_unemployment[df_unemployment['Country'] == country]
    if not unemployment_row.empty and not pd.isna(unemployment_row[col_unemployment].values[0]):
        val_unemployment = unemployment_row[col_unemployment].values[0]
    else:
        val_unemployment = None
else:
    val_unemployment = None
default_215 = row[f'{year}_215'].values[0] if not row.empty and f'{year}_215' in row.columns else 0.0
default_365 = row[f'{year}_365'].values[0] if not row.empty and f'{year}_365' in row.columns else 0.0
default_literacy = row[col_literacy].values[0] if col_literacy in row.columns else 0.0

# literacy rate slider
col_literacy = f"{year}_literacy"
val_literacy = row[col_literacy].values[0] if col_literacy in row.columns and not pd.isna(row[col_literacy].values[0]) else None
if col_literacy in df_literacy.columns:
    literacy_row = df_literacy[df_literacy['Country'] == country]
    if not literacy_row.empty and not pd.isna(literacy_row[col_literacy].values[0]):
        default_literacy = literacy_row[col_literacy].values[0]
    else:
        default_literacy = 0.0
else:
    default_literacy = 0.0

# employment rate
col_employment = f"{year}_employment"
val_employment = row[col_employment].values[0] if col_employment in row.columns and not pd.isna(row[col_employment].values[0]) else None
default_employment = val_employment if val_employment is not None else 0.0

# unemployment rate slider
col_unemployment = str(year)
default_unemployment = row[col_unemployment].values[0] if col_unemployment in row.columns else 0.0

st.sidebar.markdown("#### Simulate Policy Changes")

def initialize_slider_state(year, row):
    # Only set if not already set
    if "sim_215" not in st.session_state:
        st.session_state.sim_215 = float(row[f'{year}_215'].values[0]) if f'{year}_215' in row.columns else 0.0
    if "sim_365" not in st.session_state:
        st.session_state.sim_365 = float(row[f'{year}_365'].values[0]) if f'{year}_365' in row.columns else 0.0
    if "sim_unemployment" not in st.session_state:
        val = row[str(year)].values[0] if str(year) in row.columns else 0.0
        st.session_state.sim_unemployment = float(val * 100 if val is not None else 0.0)
    if "sim_literacy" not in st.session_state:
        val = row[f"{year}_literacy"].values[0] if f"{year}_literacy" in row.columns else 0.0
        st.session_state.sim_literacy = float(val if val is not None else 0.0)
    if "sim_employment" not in st.session_state:
        val = row[f"{year}_employment"].values[0] if f"{year}_employment" in row.columns else 0.0
        st.session_state.sim_employment = float(val if val is not None else 0.0)

initialize_slider_state(year, row)

sim_215 = st.sidebar.slider(
    f"Poverty Headcount Ratio at $2.15/day ({year})", min_value=0.0, max_value=1.0,
    value=st.session_state.sim_215, step=0.01,
    key="sim_215",
    help="Adjust to simulate changes in extreme poverty rate."
)
sim_365 = st.sidebar.slider(
    f"Poverty Headcount Ratio at $3.65/day ({year})", min_value=0.0, max_value=1.0,
    value=st.session_state.sim_365, step=0.01,
    key="sim_365",
    help="Adjust to simulate changes in moderate poverty rate."
)
sim_unemployment = st.sidebar.slider(
    f"Unemployment Rate (%) ({year})", min_value=0.0, max_value=100.0,
    value=st.session_state.sim_unemployment, step=0.1,
    key="sim_unemployment",
    help="Adjust to simulate changes in unemployment rate."
)
sim_literacy = st.sidebar.slider(
    f"Literacy Rate (%) ({year})", min_value=0.0, max_value=100.0,
    value=st.session_state.sim_literacy, step=0.1,
    key="sim_literacy",
    help="Adjust to simulate changes in literacy rate."
)
sim_employment = st.sidebar.slider(
    f"Employment Rate (%) ({year})", min_value=0.0, max_value=100.0,
    value=st.session_state.sim_employment, step=0.1,
    key="sim_employment",
    help="Adjust to simulate changes in employment-to-population ratio."
)

# if st.sidebar.button("Reset to 2025 Country Values"):
#     st.session_state.sim_215 = float(row['2025_215'].values[0]) if '2025_215' in row.columns else 0.0
#     st.session_state.sim_365 = float(row['2025_365'].values[0]) if '2025_365' in row.columns else 0.0
#     val_unemp = row['2025'].values[0] if '2025' in row.columns else 0.0
#     st.session_state.sim_unemployment = float(val_unemp * 100 if val_unemp is not None else 0.0)
#     val_lit = row['2025_literacy'].values[0] if '2025_literacy' in row.columns else 0.0
#     st.session_state.sim_literacy = float(val_lit if val_lit is not None else 0.0)
#     val_emp = row['2025_employment'].values[0] if '2025_employment' in row.columns else 0.0
#     st.session_state.sim_employment = float(val_emp if val_emp is not None else 0.0)
#     st.experimental_rerun()

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
        if col_unemployment in X.columns:
            X.at[X.index[0], col_unemployment] = sim_unemployment / 100
        if col_literacy in X.columns:
            X.at[X.index[0], col_literacy] = sim_literacy / 100

        pred = model.predict(X)[0]
        st.header("Prediction Result")
        st.caption(f"This section shows the predicted poverty risk for {country} in {year} based on socioeconomic indicators.")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 280px;">
                <span style="font-size: 8rem; font-weight: bold; color: #4cbb17;">{pred:.2%}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # third row: key indicators
    st.header("Key Poverty Indicators")
    val_215 = sim_215
    val_365 = sim_365

    col_post_tax = f"{year}"
    if col_post_tax in row.columns and not pd.isna(row[col_post_tax].values[0]):
        val_post_tax = row[col_post_tax].values[0]
    else:
        val_post_tax = None
    
    # post tax logic design
    val_post_tax = None
    if str(year) in df_post_tax.columns:
        post_tax_row = df_post_tax[df_post_tax['Country'] == country]
        if not post_tax_row.empty and not pd.isna(post_tax_row[str(year)].values[0]):
            val_post_tax = post_tax_row[str(year)].values[0]

    # unemployment logic design
    col_unemployment = str(year)
    if col_unemployment in df_unemployment.columns:
        unemployment_row = df_unemployment[df_unemployment['Country'] == country]
        if not unemployment_row.empty and not pd.isna(unemployment_row[col_unemployment].values[0]):
            val_unemployment = unemployment_row[col_unemployment].values[0]
        else:
            val_unemployment = None
    else:
        val_unemployment = None
    
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
            value=f"{val_post_tax * 100:.2f}%" if val_post_tax is not None else "N/A",
            help="Poverty Rate After Taxes and Transfers: Share of people living below the poverty line after government taxes and social transfers. Shows the impact of social protection policies."
        )
    
    # add some space between rows (indicators)
    st.markdown("<br>", unsafe_allow_html=True)  

    added_col = st.columns(3)
    with added_col[0]:
        st.metric(
            label=f"Unemployment Rate ({year})",
            value=f"{sim_unemployment:.1f}%" if sim_unemployment is not None else "N/A",
            help="Unemployment Rate: Percentage of the labor force that is jobless. Higher unemployment can increase poverty risk."
     )

    with added_col[1]:
        st.metric(
            label=f"Literacy Rate ({year})",
            value=f"{sim_literacy:.1f}%" if sim_literacy is not None else "N/A",
            help="Literacy Rate: Percentage of people aged 15 and above who can read and write."
        )

    with added_col[2]:
        st.metric(
            label=f"Employment Rate ({year})",
            value=f"{sim_employment:.1f}%" if sim_employment is not None else "N/A",
            help="Employment-to-population ratio: Percentage of working-age population that is employed."
        )
    
    # add some space
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("<br>", unsafe_allow_html=True)

    # plot country vs global average graph
    col_unemployment_name = str(year)
    indicators = {
        "Poverty Headcount Ratio at $2.15/day": col_215,
        "Poverty Headcount Ratio at $3.65/day": col_365,
        "Poverty Rate After Taxes and Transfers": col_post_tax,
        "Unemployment Rate": col_unemployment,
        "Literacy Rate": col_literacy,
        "Employment Rate": col_employment
    }

    selected_country_values = [val_215, 
                               val_365, 
                               val_post_tax if val_post_tax is not None else None,
                               val_unemployment if val_unemployment is not None else None,
                               val_literacy if val_literacy is not None else None,
                               sim_employment if sim_employment is not None else None]
    
    global_avg_values = [df[col_215].mean() if col_215 in df.columns else None,
                        df[col_365].mean() if col_365 in df.columns else None,
                        df[col_post_tax].mean() if col_post_tax in df.columns else None,
                        df[col_unemployment_name].mean() if col_unemployment_name in df.columns else None,
                        df[col_literacy].mean() if col_literacy in df.columns else None,
                        df[col_employment].mean() if col_employment in df.columns else None]

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
        "Poverty Rate After Taxes and Transfers": [str(year) for year in range(2000, 2026)],
        "Unemployment Rate": [f"{year}_unemployment" for year in range(2000, 2026)],
        "Literacy Rate": [f"{year}_literacy" for year in range(2000, 2026)],
        "Employment Rate": [f"{year}_employment" for year in range(2000, 2026)]
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
        height=412
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
        st.markdown(
            f"<h2 style='margin-bottom:0;'>{country} Poverty, Unemployment, and Literacy Trends</h2>",
            unsafe_allow_html=True
        )
        st.caption(
            f"Explore how poverty rate after taxes and transfers, unemployment rate, and literacy rate have changed in {country} from 2000 to 2025. "
            "This visualization helps you understand the interplay between education, labor market conditions, and poverty over time."
        )
        st.plotly_chart(fig_tt_trend, use_container_width=True, key="tt_trend")
        st.divider()
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

            # unemployment rate
            series_unemp = trend_tt_data["Unemployment Rate"].dropna()
            nonzero_unemp = series_unemp[series_unemp != 0]
            if not series_unemp.empty:
                max_unemp = series_unemp.max()
                max_unemp_idx = series_unemp.idxmax()
                max_unemp_year = trend_tt_data.loc[max_unemp_idx, "Year"]
                min_unemp = series_unemp.min()
                min_unemp_idx = series_unemp.idxmin()
                min_unemp_year = trend_tt_data.loc[min_unemp_idx, "Year"]
                avg_unemp = nonzero_unemp.mean() if not nonzero_unemp.empty else 0
                st.markdown(
                    f"- The **highest unemployment rate** was **{max_unemp:.2%}** in **{max_unemp_year}**."
                )
                st.markdown(
                    f"- The **lowest unemployment rate** was **{min_unemp:.2%}** in **{min_unemp_year}**."
                )
                st.markdown(
                    f"- The **average unemployment rate** (excluding 0 and missing data) is **{avg_unemp:.2%}**."
                )
            else:
                st.markdown("- Data for unemployment rate is not available for trend analysis.")

            # literacy rate
            series_lit = trend_tt_data["Literacy Rate"].dropna()
            nonzero_lit = series_lit[series_lit != 0]
            if not series_lit.empty:
                max_lit = series_lit.max()
                max_lit_idx = series_lit.idxmax()
                max_lit_year = trend_tt_data.loc[max_lit_idx, "Year"]
                min_lit = series_lit.min()
                min_lit_idx = series_lit.idxmin()
                min_lit_year = trend_tt_data.loc[min_lit_idx, "Year"]
                avg_lit = nonzero_lit.mean() if not nonzero_lit.empty else 0
                st.markdown(
                    f"- The **highest literacy rate** was **{max_lit:.2f}%** in **{max_lit_year}**."
                )
                st.markdown(
                    f"- The **lowest literacy rate** was **{min_lit:.2f}%** in **{min_lit_year}**."
                )
                st.markdown(
                    f"- The **average literacy rate** (excluding 0 and missing data) is **{avg_lit:.2f}%**."
                )
            else:
                st.markdown("- Data for literacy rate is not available for trend analysis.")

            # employment rate
            series_emp = trend_tt_data["Employment Rate"].dropna()
            nonzero_emp = series_emp[series_emp != 0]
            if not series_emp.empty:
                max_emp = series_emp.max()
                max_emp_idx = series_emp.idxmax()
                max_emp_year = trend_tt_data.loc[max_emp_idx, "Year"]
                min_emp = series_emp.min()
                min_emp_idx = series_emp.idxmin()
                min_emp_year = trend_tt_data.loc[min_emp_idx, "Year"]
                avg_emp = nonzero_emp.mean() if not nonzero_emp.empty else 0
                st.markdown(
                    f"- The **highest employment rate** was **{max_emp:.2f}%** in **{max_emp_year}**."
                )
                st.markdown(
                    f"- The **lowest employment rate** was **{min_emp:.2f}%** in **{min_emp_year}**."
                )
                st.markdown(
                    f"- The **average employment rate** (excluding 0 and missing data) is **{avg_emp:.2f}%**."
                )
            else:
                st.markdown("- Data for employment rate is not available for trend analysis.")

            # relationships
            if not series_tt.empty and not series_unemp.empty:
                corr_pov_unemp = series_tt.corr(series_unemp)
                st.markdown(
                    f"The correlation between poverty rate after taxes and transfers and unemployment rate over time is **{corr_pov_unemp:.2f}**. "
                    "A positive value suggests that higher unemployment is associated with higher poverty after taxes and transfers."
                )
            if not series_tt.empty and not series_lit.empty:
                corr_pov_lit = series_tt.corr(series_lit)
                st.markdown(
                    f"The correlation between poverty rate after taxes and transfers and literacy rate over time is **{corr_pov_lit:.2f}**. "
                    "A negative value suggests that higher literacy is associated with lower poverty after taxes and transfers."
                )
            if not series_tt.empty and not series_emp.empty:
                corr_pov_emp = series_tt.corr(series_emp)
                st.markdown(
                    f"The correlation between poverty rate after taxes and transfers and employment rate over time is **{corr_pov_emp:.2f}**. "
                    "A negative value suggests that higher employment is associated with lower poverty after taxes and transfers."
                )
            if not series_emp.empty and not series_unemp.empty:
                corr_emp_unemp = series_emp.corr(series_unemp)
                st.markdown(
                    f"The correlation between employment rate and unemployment rate over time is **{corr_emp_unemp:.2f}**. "
                    "A strong negative value suggests that higher employment is associated with lower unemployment."
                )

    with fourth_row[1]:
        st.header(f"**{country} Poverty Indicator Trends**")
        st.caption(
            f"This chart shows the historical trends of poverty indicators in {country} from 2000 to 2025. "
            "You can observe how the poverty headcount ratios have changed over time and compare their patterns."
        )

        # add some space
        st.markdown("<br>", unsafe_allow_html=True)    
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.plotly_chart(fig_trend, use_container_width=True, key="indicator_trends")
        st.divider()
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
    advice_lines = []

    # poverty headcount ratio at $2.15/day
    if val_215 is not None and global_avg_values[0] is not None:
        if val_215 > global_avg_values[0]:
            advice_lines.append(
                f"- {country} has a higher extreme poverty rate ($2.15/day) than the global average. Strengthen social safety nets and targeted poverty alleviation programs."
            )
        else:
            advice_lines.append(
                f"- {country}'s extreme poverty rate ($2.15/day) is below the global average. Maintain current policies and monitor for emerging risks."
            )

    # poverty headcount ratio $3.65/day
    if val_365 is not None and global_avg_values[1] is not None:
        if val_365 > global_avg_values[1]:
            advice_lines.append(
                f"- {country} has a higher moderate poverty rate ($3.65/day) than the global average. Focus on inclusive economic growth and job creation."
            )
        else:
            advice_lines.append(
                f"- {country}'s moderate poverty rate ($3.65/day) is below the global average. Maintain efforts to support vulnerable groups."
           )

    # post-tax
    if val_post_tax is not None and global_avg_values[2] is not None:
        if val_post_tax > global_avg_values[2]:
            advice_lines.append(
                "- The poverty rate after taxes and transfers is above the global average. Review and enhance social protection and tax policies."
            )
        else:
            advice_lines.append(
                "- Social protection policies are effective compared to global average. Continue to invest in inclusive programs."
            )

    # unemployment rate
    if sim_unemployment is not None and global_avg_values[3] is not None:
        if sim_unemployment > global_avg_values[3]:
            advice_lines.append(
               "- Unemployment rate is above the global average. Invest in job creation, skills training, and labor market reforms."
           )
        else:
            advice_lines.append(
                "- Unemployment rate is below the global average. Maintain labor market stability and support workforce development."
            )

    # literacy rate
    if sim_literacy is not None and global_avg_values[4] is not None:
        if sim_literacy < global_avg_values[4]:
            advice_lines.append(
                "- Literacy rate is below the global average. Invest in education access, quality, and adult literacy programs."
            )
        else:
            advice_lines.append(
                "- Literacy rate is above the global average. Continue supporting education and lifelong learning."
            )

    # employment rate
    if sim_employment is not None and global_avg_values[5] is not None:
        if sim_employment < global_avg_values[5]:
            advice_lines.append(
               "- Employment rate is below the global average. Promote job opportunities and reduce barriers to employment."
        )
        else:
            advice_lines.append(
               "- Employment rate is above the global average. Maintain policies that support high labor force participation."
            )

    # trend poverty
    series_215 = trend_data["Poverty Headcount Ratio at $2.15/day"].dropna()
    if not series_215.empty and series_215.iloc[-1] > series_215.iloc[0]:
        advice_lines.append(
            "- The extreme poverty rate ($2.15/day) has increased over time. Investigate causes and strengthen poverty reduction strategies."
        )
    elif not series_215.empty and series_215.iloc[-1] < series_215.iloc[0]:
        advice_lines.append(
            "- The extreme poverty rate ($2.15/day) has decreased over time. Continue successful interventions and monitor progress."
        )

    series_365 = trend_data["Poverty Headcount Ratio at $3.65/day"].dropna()
    if not series_365.empty and series_365.iloc[-1] > series_365.iloc[0]:
        advice_lines.append(
            "- The moderate poverty rate ($3.65/day) has increased over time. Investigate causes and strengthen poverty reduction strategies."
        )
    elif not series_365.empty and series_365.iloc[-1] < series_365.iloc[0]:
        advice_lines.append(
            "- The moderate poverty rate ($3.65/day) has decreased over time. Continue successful interventions and monitor progress."
        )

    # trend unemployment
    series_unemp = trend_tt_data["Unemployment Rate"].dropna()
    if not series_unemp.empty and series_unemp.iloc[-1] > series_unemp.iloc[0]:
        advice_lines.append(
            "- Unemployment rate has increased over time. Expand job creation initiatives and workforce training."
        )
    elif not series_unemp.empty and series_unemp.iloc[-1] < series_unemp.iloc[0]:
        advice_lines.append(
            "- Unemployment rate has decreased over time. Maintain effective labor market policies."
        )

    # trend literacy
    series_lit = trend_tt_data["Literacy Rate"].dropna()
    if not series_lit.empty and series_lit.iloc[-1] > series_lit.iloc[0]:
        advice_lines.append(
            "- Literacy rate has improved over time. Continue investing in education and literacy programs."
        )
    elif not series_lit.empty and series_lit.iloc[-1] < series_lit.iloc[0]:
        advice_lines.append(
           "- Literacy rate has declined over time. Investigate barriers and invest in education access."
        )

    # trend employment
    series_emp = trend_tt_data["Employment Rate"].dropna()
    if not series_emp.empty and series_emp.iloc[-1] > series_emp.iloc[0]:
        advice_lines.append(
            "- Employment rate has increased over time. Maintain policies that support job growth."
        )
    elif not series_emp.empty and series_emp.iloc[-1] < series_emp.iloc[0]:
        advice_lines.append(
           "- Employment rate has decreased over time. Address barriers to employment and promote workforce participation."
        )

    # relationship
    if not series_tt.empty and not series_unemp.empty:
        corr_pov_unemp = series_tt.corr(series_unemp)
        if corr_pov_unemp > 0.3:
            advice_lines.append(
                "- High correlation between poverty and unemployment suggests that job creation can help reduce poverty."
            )
    if not series_tt.empty and not series_lit.empty:
        corr_pov_lit = series_tt.corr(series_lit)
        if corr_pov_lit < -0.3:
            advice_lines.append(
                "- Strong negative correlation between poverty and literacy suggests that improving education can lower poverty."
            )
    if not series_tt.empty and not series_emp.empty:
        corr_pov_emp = series_tt.corr(series_emp)
        if corr_pov_emp < -0.3:
            advice_lines.append(
                "- Strong negative correlation between poverty and employment rate suggests that increasing employment can reduce poverty."
            )
    if not series_emp.empty and not series_unemp.empty:
        corr_emp_unemp = series_emp.corr(series_unemp)
        if corr_emp_unemp < -0.3:
            advice_lines.append(
                "- Strong negative correlation between employment and unemployment rates confirms that boosting employment helps lower unemployment."
            )

    # general advice
    advice_lines.append(
        "- Consider investing in education, healthcare, and social protection to further reduce poverty and improve well-being."
    )
    st.markdown("\n".join(advice_lines))

    # report data
    # st.subheader("Downloadable Report")

    # def create_pdf(country, pred, row, val_post_tax, desc_lines, insights, advice_lines, fig, fig_trend):
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_auto_page_break(auto=True, margin=15)
    #     # title
    #     pdf.set_font("Arial", 'B', 18)
    #     pdf.set_text_color(0, 70, 140)
    #     pdf.cell(0, 15, f"PovertyLens Report: {country}", ln=True, align='C')
    #     pdf.set_text_color(0, 0, 0)
    #     # section separator
    #     pdf.set_draw_color(0, 70, 140)
    #     pdf.set_line_width(1)
    #     pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    #     pdf.ln(5)
    #     # prediction
    #     pdf.set_font("Arial", 'B', 14)
    #     pdf.cell(0, 10, "Prediction Result", ln=True)
    #     pdf.set_font("Arial", '', 12)
    #     pdf.set_text_color(34, 139, 34)
    #     pdf.cell(0, 10, f"Poverty Risk (2025): {pred:.2%}", ln=True)
    #     pdf.set_text_color(0, 0, 0)
    #     pdf.ln(5)
    #     # key indicators
    #     pdf.set_font("Arial", 'B', 14)
    #     pdf.cell(0, 10, "Key Poverty Indicators", ln=True)
    #     pdf.set_font("Arial", '', 12)
    #     pdf.cell(0, 8, f"- Poverty Headcount Ratio at $2.15/day (2025): {row['2025_215'].values[0]:.2%}", ln=True)
    #     pdf.cell(0, 8, f"- Poverty Headcount Ratio at $3.65/day (2025): {row['2025_365'].values[0]:.2%}", ln=True)
    #     pdf.cell(0, 8, f"- Poverty Rate After Taxes and Transfers (2025): {val_post_tax if val_post_tax else 'Data not available'}", ln=True)
    #     pdf.ln(5)
    #     # section separator
    #     pdf.set_draw_color(220, 220, 220)
    #     pdf.set_line_width(0.5)
    #     pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    #     pdf.ln(5)
    #     # insights
    #     pdf.set_font("Arial", 'B', 14)
    #     pdf.cell(0, 10, "Insights", ln=True)
    #     pdf.set_font("Arial", '', 12)
    #     for line in desc_lines:
    #         pdf.multi_cell(0, 8, f"- {line}")
    #     pdf.ln(3)
    #     # trend insights
    #     pdf.set_font("Arial", 'B', 14)
    #     pdf.cell(0, 10, "Trend Insights", ln=True)
    #     pdf.set_font("Arial", '', 12)
    #     for line in insights:
    #         pdf.multi_cell(0, 8, line)
    #     pdf.ln(3)
    #     # smart a dvice
    #     pdf.set_font("Arial", 'B', 14)
    #     pdf.cell(0, 10, "Smart Advice", ln=True)
    #     pdf.set_font("Arial", '', 12)
    #     for line in advice_lines:
    #         pdf.multi_cell(0, 8, line)
    #     pdf.ln(5)
    #     # section separator
    #     pdf.set_draw_color(0, 70, 140)
    #     pdf.set_line_width(1)
    #     pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    #     pdf.ln(5)
    #     # save charts as images and add to PDF
    #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile1:
    #         fig.write_image(tmpfile1.name)
    #         pdf.image(tmpfile1.name, w=180)
    #     pdf.ln(5)
    #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile2:
    #         fig_trend.write_image(tmpfile2.name)
    #         pdf.image(tmpfile2.name, w=180)
    #     pdf.ln(5)
    #     # footer
    #     pdf.set_y(-20)
    #     pdf.set_font("Arial", 'I', 10)
    #     pdf.set_text_color(120, 120, 120)
    #     pdf.cell(0, 10, "Generated by PovertyLens | github.com/priscillahandoyo/poverty-lens", 0, 0, 'C')
    #     # save PDF to bytes
    #     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
    #         pdf.output(pdf_file.name)
    #         pdf_file.seek(0)
    #         pdf_bytes = pdf_file.read()
    #     return pdf_bytes

    # # button to download PDF
    # if st.button("Download Full Report as PDF"):
    #     val_post_tax_str = f"{val_post_tax:.2%}" if val_post_tax is not None else "Data not available"
    #     pdf_bytes = create_pdf(
    #         country, pred, row, val_post_tax_str, desc_lines, insights, advice_lines, fig, fig_trend
    #     )
    #     st.download_button(
    #         label="Click to Download PDF",
    #         data=pdf_bytes,
    #         file_name=f"{country}_poverty_report.pdf",
    #         mime="application/pdf"
    #     )

else:
    st.warning("Country data not found.")
