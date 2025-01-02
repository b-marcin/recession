import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Enhanced Recession Risk Dashboard", page_icon="📈", layout="wide")

# Initialize FRED API
if 'FRED_API_KEY' not in st.secrets:
    st.error("Please set your FRED API key in the secrets management.")
    st.stop()

fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# Enhanced data fetching with retry logic
@st.cache_data(ttl=86400)
def get_fred_data(series_id, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            return pd.DataFrame(series, columns=[series_id])
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Error fetching {series_id}: {str(e)}")
                return pd.DataFrame()
            continue

def calculate_yield_curve_probability(spread):
    if spread <= -0.5:
        return "High Risk ⚠️"
    elif spread <= 0:
        return "Moderate Risk ⚡"
    else:
        return "Low Risk ✅"

def calculate_unemployment_warning(current, rolling_mean):
    if current > rolling_mean * 1.1:  # 10% above moving average
        return "High Risk ⚠️"
    elif current > rolling_mean * 1.05:  # 5% above moving average
        return "Moderate Risk ⚡"
    return "Low Risk ✅"

def check_industrial_production_decline(df):
    if len(df) < 6:
        return "Insufficient Data"
    
    six_month_change = df[df.columns[0]].pct_change(periods=6).iloc[-1] * 100
    
    if six_month_change < -2:
        return "High Risk ⚠️"
    elif six_month_change < 0:
        return "Moderate Risk ⚡"
    return "Low Risk ✅"

# Set date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)

# Define all indicators
indicators = {
    'T10Y2Y': 'Treasury Yield Spread (10Y-2Y)',
    'UNRATE': 'Unemployment Rate',
    'INDPRO': 'Industrial Production',
    'KCFSI': 'Kansas City Fed Financial Stress Index',
    'USREC': 'NBER Recession Indicator',
    'USSLIND': 'Leading Index',
    'BAA10Y': 'Corporate Bond Spread'
}

# Fetch all indicators
indicator_data = {}
for series_id in indicators.keys():
    data = get_fred_data(series_id, start_date, end_date)
    if not data.empty:
        indicator_data[series_id] = data

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Yield Curve Plot
    st.subheader("📊 Treasury Yield Spread (10Y-2Y)")
    if 'T10Y2Y' in indicator_data:
        yield_curve = indicator_data['T10Y2Y']
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
        fig.add_hrect(y0=-0.5, y1=-5, 
                     fillcolor="red", opacity=0.1, 
                     line_width=0, name="High Risk Zone")
        fig.add_hrect(y0=0, y1=-0.5, 
                     fillcolor="yellow", opacity=0.1, 
                     line_width=0, name="Moderate Risk Zone")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Create tabs for different indicators
    tab1, tab2, tab3, tab4 = st.tabs(["Unemployment", "Industrial Production", "Leading Index", "Corporate Spread"])
    
    with tab1:
        if 'UNRATE' in indicator_data:
            unemployment = indicator_data['UNRATE']
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
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'INDPRO' in indicator_data:
            production = indicator_data['INDPRO']
            warning_status = check_industrial_production_decline(production)
            st.markdown(f"**Current Status: {warning_status}**")
            
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
            
            # Calculate and add 6-month change
            production['Change'] = production['INDPRO'].pct_change(periods=6) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=production.index,
                    y=production['Change'],
                    name='6-Month Change (%)',
                    line=dict(color='orange', dash='dash')
                ),
                secondary_y=True
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'USSLIND' in indicator_data:
            leading_index = indicator_data['USSLIND']
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=leading_index.index,
                    y=leading_index['USSLIND'],
                    name='Leading Index',
                    line=dict(color='purple')
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=leading_index.index,
                    y=leading_index['USSLIND'].rolling(window=3).mean(),
                    name='3-Month Average',
                    line=dict(color='gray', dash='dash')
                )
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'BAA10Y' in indicator_data:
            bond_spread = indicator_data['BAA10Y']
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bond_spread.index,
                    y=bond_spread['BAA10Y'],
                    name='BAA-10Y Spread',
                    line=dict(color='brown')
                )
            )
            fig.add_hrect(
                y0=3, y1=5,
                fillcolor="red", opacity=0.1,
                line_width=0, name="High Risk Zone"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with col2:
    # Financial Stress Index
    st.subheader("💹 Financial Stress Index")
    if 'KCFSI' in indicator_data:
        stress_index = indicator_data['KCFSI']
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
        fig.add_hrect(y0=1, y1=5,
                     fillcolor="red", opacity=0.1,
                     line_width=0, name="High Risk Zone")
        fig.add_hrect(y0=0, y1=1,
                     fillcolor="yellow", opacity=0.1,
                     line_width=0, name="Moderate Risk Zone")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Risk Assessment
    st.subheader("🚨 Risk Assessment")
    
    risk_indicators = []
    
    if 'T10Y2Y' in indicator_data:
        risk_indicators.append({
            "Indicator": "Yield Curve",
            "Status": calculate_yield_curve_probability(indicator_data['T10Y2Y']['T10Y2Y'].iloc[-1])
        })
    
    if 'UNRATE' in indicator_data:
        unemployment = indicator_data['UNRATE']
        ma12 = unemployment['UNRATE'].rolling(12).mean()
        risk_indicators.append({
            "Indicator": "Unemployment",
            "Status": calculate_unemployment_warning(
                unemployment['UNRATE'].iloc[-1],
                ma12.iloc[-1]
            )
        })
    
    if 'INDPRO' in indicator_data:
        risk_indicators.append({
            "Indicator": "Industrial Production",
            "Status": check_industrial_production_decline(indicator_data['INDPRO'])
        })
    
    if 'KCFSI' in indicator_data:
        risk_indicators.append({
            "Indicator": "Financial Stress",
            "Status": stress_warning
        })
    
    if risk_indicators:
        risk_df = pd.DataFrame(risk_indicators)
        st.table(risk_df)
        
        high_risk_count = sum(1 for x in risk_indicators if "High Risk" in x["Status"])
        moderate_risk_count = sum(1 for x in risk_indicators if "Moderate Risk" in x["Status"])
        
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
st.markdown("""
Data source: Federal Reserve Economic Data (FRED)  
Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))