import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Enhanced Recession Risk Dashboard", page_icon="📈", layout="wide")

# Title and description
st.title("🌡️ Enhanced Recession Risk Dashboard")
st.markdown("""
This dashboard combines multiple leading indicators to provide an early warning system for recession risks.
Indicators are weighted based on their historical predictive power.
""")

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

# Define indicator weights and thresholds
INDICATOR_WEIGHTS = {
    'T10Y2Y': 0.25,    # Yield curve (strongest predictor)
    'USSLIND': 0.20,   # Leading Index
    'UNRATE': 0.15,    # Unemployment
    'BAA10Y': 0.15,    # Corporate Bond Spread
    'INDPRO': 0.15,    # Industrial Production
    'KCFSI': 0.10     # Financial Stress
}

# Calculate composite risk score (0-100)
def calculate_composite_risk(indicators_data):
    scores = {}
    
    # Yield Curve Score
    if 'T10Y2Y' in indicators_data:
        value = indicators_data['T10Y2Y'].iloc[-1]
        scores['T10Y2Y'] = max(0, min(100, ((-value + 0.5) * 100)))
    
    # Leading Index Score
    if 'USSLIND' in indicators_data:
        value = indicators_data['USSLIND'].iloc[-1]
        mom_change = indicators_data['USSLIND'].pct_change(periods=3).iloc[-1] * 100
        scores['USSLIND'] = max(0, min(100, (-mom_change + 1) * 50))
    
    # Unemployment Score
    if 'UNRATE' in indicators_data:
        current = indicators_data['UNRATE'].iloc[-1]
        ma12 = indicators_data['UNRATE'].rolling(12).mean().iloc[-1]
        scores['UNRATE'] = max(0, min(100, ((current/ma12 - 1) * 500)))
    
    # Corporate Bond Spread Score
    if 'BAA10Y' in indicators_data:
        value = indicators_data['BAA10Y'].iloc[-1]
        scores['BAA10Y'] = max(0, min(100, (value - 2) * 33.33))
    
    # Industrial Production Score
    if 'INDPRO' in indicators_data:
        mom_change = indicators_data['INDPRO'].pct_change(periods=6).iloc[-1] * 100
        scores['INDPRO'] = max(0, min(100, (-mom_change + 2) * 33.33))
    
    # Financial Stress Score
    if 'KCFSI' in indicators_data:
        value = indicators_data['KCFSI'].iloc[-1]
        scores['KCFSI'] = max(0, min(100, (value + 1) * 50))
    
    # Calculate weighted average
    weighted_score = sum(scores[k] * INDICATOR_WEIGHTS[k] for k in scores.keys())
    
    return weighted_score, scores

# Set date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)

# Fetch all indicators
indicators_data = {}
for series_id in INDICATOR_WEIGHTS.keys():
    data = get_fred_data(series_id, start_date, end_date)
    if not data.empty:
        indicators_data[series_id] = data[series_id]

# Calculate composite risk
if indicators_data:
    composite_score, individual_scores = calculate_composite_risk(indicators_data)
else:
    st.error("Unable to fetch required data.")
    st.stop()

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    # Composite Risk Gauge
    st.subheader("📊 Composite Recession Risk")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = composite_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        },
        title = {'text': "Recession Risk Score"}
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Leading Index and Corporate Spread
    st.subheader("📈 Leading Economic Indicators")
    
    # Create tabs for different indicators
    tab1, tab2, tab3 = st.tabs(["Leading Index", "Corporate Spread", "Industrial Production"])
    
    with tab1:
        if 'USSLIND' in indicators_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=indicators_data['USSLIND'].index,
                y=indicators_data['USSLIND'],
                name='Leading Index',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=indicators_data['USSLIND'].index,
                y=indicators_data['USSLIND'].rolling(window=3).mean(),
                name='3-Month Average',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(title_text="Conference Board Leading Economic Index")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'BAA10Y' in indicators_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=indicators_data['BAA10Y'].index,
                y=indicators_data['BAA10Y'],
                name='BAA-10Y Spread',
                line=dict(color='purple')
            ))
            fig.add_hrect(y0=3, y1=5, 
                         fillcolor="red", opacity=0.1,
                         line_width=0, name="High Risk Zone")
            fig.update_layout(title_text="Corporate Bond Spread (BAA-10Y)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'INDPRO' in indicators_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=indicators_data['INDPRO'].index,
                y=indicators_data['INDPRO'].pct_change(periods=6) * 100,
                name='6-Month Change',
                line=dict(color='green')
            ))
            fig.add_hrect(y0=-2, y1=-10,
                         fillcolor="red", opacity=0.1,
                         line_width=0, name="Contraction Zone")
            fig.update_layout(title_text="Industrial Production (6-Month Change)")
            st.plotly_chart(fig, use_container_width=True)

with col2:
    # Risk Breakdown
    st.subheader("🔍 Risk Breakdown")
    
    risk_data = []
    for indicator, weight in INDICATOR_WEIGHTS.items():
        if indicator in individual_scores:
            risk_data.append({
                "Indicator": indicator,
                "Risk Score": round(individual_scores[indicator], 1),
                "Weight": f"{weight*100}%",
                "Contribution": round(individual_scores[indicator] * weight, 1)
            })
    
    risk_df = pd.DataFrame(risk_data)
    risk_df = risk_df.sort_values("Contribution", ascending=False)
    
    # Color-code the risk scores
    def color_risk(val):
        if isinstance(val, (int, float)):
            if val >= 66:
                return 'background-color: rgba(255,0,0,0.2)'
            elif val >= 33:
                return 'background-color: rgba(255,255,0,0.2)'
            else:
                return 'background-color: rgba(0,255,0,0.2)'
        return ''
    
    st.dataframe(risk_df.style.applymap(color_risk, subset=['Risk Score']))
    
    # Historical Comparison
    st.subheader("📊 Historical Context")
    
    # Calculate historical risk scores
    historical_scores = pd.DataFrame(index=indicators_data['T10Y2Y'].index)
    historical_scores['Risk Score'] = 0
    
    for date in historical_scores.index:
        point_data = {k: v[v.index <= date].iloc[-1] for k, v in indicators_data.items()}
        score, _ = calculate_composite_risk(point_data)
        historical_scores.loc[date, 'Risk Score'] = score
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_scores.index,
        y=historical_scores['Risk Score'],
        name='Historical Risk',
        line=dict(color='blue')
    ))
    
    # Add NBER recession shading if available
    recession_data = get_fred_data('USREC', start_date, end_date)
    if not recession_data.empty:
        recession_periods = []
        in_recession = False
        start = None
        
        for date, value in recession_data.iterrows():
            if value.iloc[0] == 1 and not in_recession:
                start = date
                in_recession = True
            elif value.iloc[0] == 0 and in_recession:
                recession_periods.append((start, date))
                in_recession = False
        
        for start, end in recession_periods:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0
            )
    
    fig.update_layout(
        title_text="Historical Risk Score",
        yaxis_title="Risk Score",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Current Risk Assessment
    risk_level = "High" if composite_score >= 66 else "Moderate" if composite_score >= 33 else "Low"
    risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Moderate" else "green"
    
    st.markdown(f"""
    ### Current Risk Assessment
    
    **Overall Risk Level:** ::{risk_color}[{risk_level}]
    
    **Composite Score:** {composite_score:.1f}/100
    
    **Key Contributors:**
    {risk_df.iloc[0]['Indicator']}: {risk_df.iloc[0]['Contribution']:.1f} points
    {risk_df.iloc[1]['Indicator']}: {risk_df.iloc[1]['Contribution']:.1f} points
    """)

# Footer
st.markdown("---")
st.markdown("""
Data source: Federal Reserve Economic Data (FRED)  
Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))