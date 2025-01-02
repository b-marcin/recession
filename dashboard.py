import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Recession Indicators Dashboard",
    page_icon="📈",
    layout="wide"
)

# Add title and description
st.title("🌡️ Economic Recession Indicators Dashboard")
st.markdown("""
This dashboard tracks key economic indicators that are commonly used to predict or identify recessions.
Data is sourced from FRED (Federal Reserve Economic Data).
""")

# Function to fetch FRED data
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_fred_data(series_id, start_date, end_date):
    try:
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {series_id}: {str(e)}")
        return pd.DataFrame()

# Set date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)  # 10 years of data

# Define indicators
indicators = {
    'T10Y2Y': 'Treasury Yield Spread (10Y-2Y)',
    'UNRATE': 'Unemployment Rate',
    'INDPRO': 'Industrial Production',
    'PAYEMS': 'Total Nonfarm Payrolls',
    'KCFSI': 'Kansas City Fed Financial Stress Index',
    'USREC': 'NBER Recession Indicator'
}

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Yield Curve Plot
    st.subheader("📊 Treasury Yield Spread (10Y-2Y)")
    yield_curve = get_fred_data('T10Y2Y', start_date, end_date)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yield_curve.index,
            y=yield_curve['T10Y2Y'],
            name='Spread',
            line=dict(color='blue')
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Treasury Yield Spread (10Y-2Y)",
        xaxis_title="Date",
        yaxis_title="Spread (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Unemployment and Industrial Production
    st.subheader("📈 Economic Activity Indicators")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Unemployment Rate", "Industrial Production"])
    
    with tab1:
        unemployment = get_fred_data('UNRATE', start_date, end_date)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=unemployment.index,
                y=unemployment['UNRATE'],
                name='Unemployment Rate',
                line=dict(color='red')
            )
        )
        fig.update_layout(
            height=400,
            title_text="Unemployment Rate",
            xaxis_title="Date",
            yaxis_title="Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        production = get_fred_data('INDPRO', start_date, end_date)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=production.index,
                y=production['INDPRO'],
                name='Industrial Production',
                line=dict(color='green')
            )
        )
        fig.update_layout(
            height=400,
            title_text="Industrial Production Index",
            xaxis_title="Date",
            yaxis_title="Index (2017=100)"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Financial Stress Index
    st.subheader("💹 Financial Stress Index")
    stress_index = get_fred_data('KCFSI', start_date, end_date)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stress_index.index,
            y=stress_index['KCFSI'],
            name='Stress Index',
            line=dict(color='purple')
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=300,
        title_text="Kansas City Fed Financial Stress Index",
        xaxis_title="Date",
        yaxis_title="Index"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add key statistics
    st.subheader("📊 Key Statistics")
    
    # Calculate latest values and changes
    for series_id, name in indicators.items():
        if series_id != 'USREC':  # Skip recession indicator for this section
            data = get_fred_data(series_id, start_date, end_date)
            if not data.empty:
                latest_value = data.iloc[-1].values[0]
                previous_value = data.iloc[-2].values[0]
                change = latest_value - previous_value
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=name,
                        value=f"{latest_value:.2f}",
                        delta=f"{change:.2f}"
                    )

# Add footer with data source
st.markdown("---")
st.markdown("Data source: Federal Reserve Economic Data (FRED)")
st.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
