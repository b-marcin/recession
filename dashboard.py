import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
Data is sourced from FRED (Federal Reserve Economic Data). Warning indicators are based on historical patterns.
""")

# Initialize FRED API
if 'FRED_API_KEY' not in st.secrets:
    st.error("Please set your FRED API key in the secrets management.")
    st.stop()

fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# Function to fetch FRED data
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_fred_data(series_id, start_date, end_date):
    try:
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return pd.DataFrame(series, columns=[series_id])
    except Exception as e:
        st.error(f"Error fetching data for {series_id}: {str(e)}")
        return pd.DataFrame()

# Function to calculate recession probability based on yield curve
def calculate_yield_curve_probability(spread):
    if spread <= -0.5:
        return "High Risk ⚠️"
    elif spread <= 0:
        return "Moderate Risk ⚡"
    else:
        return "Low Risk ✅"

# Function to calculate unemployment rate warning
def calculate_unemployment_warning(current, rolling_mean):
    if current > rolling_mean * 1.1:  # 10% above moving average
        return "High Risk ⚠️"
    elif current > rolling_mean * 1.05:  # 5% above moving average
        return "Moderate Risk ⚡"
    return "Low Risk ✅"

# Function to check industrial production decline
def check_industrial_production_decline(df):
    if len(df) < 6:
        return "Insufficient Data"
    
    latest_value = df.iloc[-1].values[0]
    six_months_ago = df.iloc[-6].values[0]
    six_month_change = (latest_value - six_months_ago) / six_months_ago * 100
    
    if six_month_change < -2:
        return "High Risk ⚠️"
    elif six_month_change < 0:
        return "Moderate Risk ⚡"
    return "Low Risk ✅"

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
    
    if not yield_curve.empty:
        current_spread = yield_curve['T10Y2Y'].iloc[-1]
        recession_probability = calculate_yield_curve_probability(current_spread)
        
        st.markdown(f"**Current Status: {recession_probability}**")
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=yield_curve.index,
                y=yield_curve['T10Y2Y'],
                name='Spread',
                line=dict(color='blue')
            )
        )
        # Add warning zones
        fig.add_hrect(y0=-0.5, y1=-5, 
                     fillcolor="red", opacity=0.1, 
                     line_width=0, name="High Risk Zone")
        fig.add_hrect(y0=0, y1=-0.5, 
                     fillcolor="yellow", opacity=0.1, 
                     line_width=0, name="Moderate Risk Zone")
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
        if not unemployment.empty:
            # Calculate 12-month moving average
            unemployment['MA12'] = unemployment['UNRATE'].rolling(window=12).mean()
            current_rate = unemployment['UNRATE'].iloc[-1]
            moving_avg = unemployment['MA12'].iloc[-1]
            warning_status = calculate_unemployment_warning(current_rate, moving_avg)
            
            st.markdown(f"**Current Status: {warning_status}**")
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=unemployment.index,
                    y=unemployment['UNRATE'],
                    name='Unemployment Rate',
                    line=dict(color='red')
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=unemployment.index,
                    y=unemployment['MA12'],
                    name='12-month Moving Average',
                    line=dict(color='gray', dash='dash')
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
        if not production.empty:
            warning_status = check_industrial_production_decline(production)
            st.markdown(f"**Current Status: {warning_status}**")
            
            # Calculate 6-month rolling percentage change
            production['6M_Change'] = production['INDPRO'].pct_change(periods=6) * 100
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=production.index,
                    y=production['INDPRO'],
                    name='Industrial Production',
                    line=dict(color='green')
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=production.index,
                    y=production['6M_Change'],
                    name='6-Month Change (%)',
                    line=dict(color='orange', dash='dash')
                ),
                secondary_y=True
            )
            
            # Add warning zones for 6-month change
            fig.add_hrect(secondary_y=True,
                         y0=-2, y1=-10,
                         fillcolor="red", opacity=0.1,
                         line_width=0, name="High Risk Zone")
            fig.add_hrect(secondary_y=True,
                         y0=0, y1=-2,
                         fillcolor="yellow", opacity=0.1,
                         line_width=0, name="Moderate Risk Zone")
            
            fig.update_layout(
                height=400,
                title_text="Industrial Production Index",
                showlegend=True
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Index (2017=100)", secondary_y=False)
            fig.update_yaxes(title_text="6-Month Change (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)

with col2:
    # Financial Stress Index
    st.subheader("💹 Financial Stress Index")
    stress_index = get_fred_data('KCFSI', start_date, end_date)
    
    if not stress_index.empty:
        current_stress = stress_index['KCFSI'].iloc[-1]
        stress_warning = "High Risk ⚠️" if current_stress > 1 else "Moderate Risk ⚡" if current_stress > 0 else "Low Risk ✅"
        
        st.markdown(f"**Current Status: {stress_warning}**")
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=stress_index.index,
                y=stress_index['KCFSI'],
                name='Stress Index',
                line=dict(color='purple')
            )
        )
        # Add warning zones
        fig.add_hrect(y0=1, y1=5,
                     fillcolor="red", opacity=0.1,
                     line_width=0, name="High Risk Zone")
        fig.add_hrect(y0=0, y1=1,
                     fillcolor="yellow", opacity=0.1,
                     line_width=0, name="Moderate Risk Zone")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=300,
            title_text="Kansas City Fed Financial Stress Index",
            xaxis_title="Date",
            yaxis_title="Index"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Add recession risk summary
    st.subheader("🚨 Recession Risk Summary")
    
    # Create summary metrics for each indicator
    indicators_status = []
    
    if not yield_curve.empty:
        indicators_status.append({
            "Indicator": "Yield Curve",
            "Status": calculate_yield_curve_probability(yield_curve['T10Y2Y'].iloc[-1])
        })
    
    if not unemployment.empty:
        indicators_status.append({
            "Indicator": "Unemployment",
            "Status": calculate_unemployment_warning(
                unemployment['UNRATE'].iloc[-1],
                unemployment['MA12'].iloc[-1]
            )
        })
    
    if not production.empty:
        indicators_status.append({
            "Indicator": "Industrial Production",
            "Status": check_industrial_production_decline(production)
        })
    
    if not stress_index.empty:
        indicators_status.append({
            "Indicator": "Financial Stress",
            "Status": stress_warning
        })
    
    # Display summary table
    if indicators_status:
        status_df = pd.DataFrame(indicators_status)
        st.table(status_df)
        
        # Calculate overall risk level
        high_risk_count = sum(1 for x in indicators_status if "High Risk" in x["Status"])
        moderate_risk_count = sum(1 for x in indicators_status if "Moderate Risk" in x["Status"])
        
        if high_risk_count >= 2:
            overall_status = "High Risk ⚠️"
            status_color = "red"
        elif high_risk_count + moderate_risk_count >= 2:
            overall_status = "Moderate Risk ⚡"
            status_color = "orange"
        else:
            overall_status = "Low Risk ✅"
            status_color = "green"
        
        st.markdown(f"### Overall Risk Level: ::{status_color}[{overall_status}]")

# Add footer with data source
st.markdown("---")
st.markdown("Data source: Federal Reserve Economic Data (FRED)")
st.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))