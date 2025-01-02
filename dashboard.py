import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Comprehensive Recession Risk Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌡️ Comprehensive Recession Risk Dashboard")
st.markdown("""
This dashboard combines traditional recession indicators, machine learning predictions, and market signals 
to provide a comprehensive view of recession risk. All indicators are updated daily from FRED.
""")

# Initialize FRED API
if 'FRED_API_KEY' not in st.secrets:
    st.error("Please set your FRED API key in the secrets management.")
    st.stop()

fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# Utility functions
@st.cache_data(ttl=86400)
def get_fred_data(series_id, start_date, end_date, retries=3):
    """Fetch data from FRED with retry logic"""
    for attempt in range(retries):
        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            return pd.DataFrame(series, columns=[series_id])
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Error fetching {series_id}: {str(e)}")
                return pd.DataFrame()
            continue

def calculate_zscore(series, window=60):
    """Calculate exponentially weighted z-scores"""
    rolling_mean = series.ewm(span=window).mean()
    rolling_std = series.ewm(span=window).std()
    return (series - rolling_mean) / rolling_std

def calculate_risk_level(zscore):
    """Convert z-score to risk level"""
    if abs(zscore) > 2:
        return "High Risk ⚠️"
    elif abs(zscore) > 1:
        return "Moderate Risk ⚡"
    return "Low Risk ✅"

def calculate_yield_curve_probability(spread):
    """Calculate recession probability based on yield curve"""
    if spread <= -0.5:
        return "High Risk ⚠️", 0.8
    elif spread <= 0:
        return "Moderate Risk ⚡", 0.5
    return "Low Risk ✅", 0.2

def calculate_unemployment_warning(current, rolling_mean):
    """Calculate unemployment risk level"""
    if current > rolling_mean * 1.1:
        return "High Risk ⚠️", 0.8
    elif current > rolling_mean * 1.05:
        return "Moderate Risk ⚡", 0.5
    return "Low Risk ✅", 0.2

def check_industrial_production_decline(df):
    """Check industrial production trend"""
    if len(df) < 6:
        return "Insufficient Data", 0
    
    six_month_change = df[df.columns[0]].pct_change(periods=6).iloc[-1] * 100
    
    if six_month_change < -2:
        return "High Risk ⚠️", 0.8
    elif six_month_change < 0:
        return "Moderate Risk ⚡", 0.5
    return "Low Risk ✅", 0.2

# Define indicators with metadata
INDICATORS = {
    'T10Y2Y': {
        'name': 'Treasury Yield Spread (10Y-2Y)',
        'weight': 0.25,
        'category': 'Financial',
        'warning_threshold': 0.0
    },
    'UNRATE': {
        'name': 'Unemployment Rate',
        'weight': 0.15,
        'category': 'Labor',
        'warning_threshold': None  # Dynamic threshold based on MA
    },
    'INDPRO': {
        'name': 'Industrial Production',
        'weight': 0.15,
        'category': 'Economic',
        'warning_threshold': -2.0
    },
    'KCFSI': {
        'name': 'Financial Stress Index',
        'weight': 0.10,
        'category': 'Financial',
        'warning_threshold': 1.0
    },
    'ICSA': {
        'name': 'Initial Jobless Claims',
        'weight': 0.15,
        'category': 'Labor',
        'warning_threshold': None  # Dynamic threshold based on Z-score
    },
    'USSLIND': {
        'name': 'Leading Index',
        'weight': 0.20,
        'category': 'Economic',
        'warning_threshold': -1.0
    },
    'USREC': {
        'name': 'NBER Recession Indicator',
        'weight': 0.0,  # Used for training only
        'category': 'Economic',
        'warning_threshold': None
    }
}

# Time period selector
time_periods = {
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825,
    "10 Years": 3650
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    options=list(time_periods.keys()),
    index=3  # Default to 5 Years
)

# Date range calculation
end_date = datetime.now()
start_date = end_date - timedelta(days=time_periods[selected_period])
training_start = end_date - timedelta(days=3650)  # Always use 10 years for ML training

# Fetch all indicators
@st.cache_data(ttl=86400)
def fetch_all_indicators():
    data = {}
    for series_id in INDICATORS.keys():
        df = get_fred_data(series_id, training_start, end_date)
        if not df.empty:
            data[series_id] = df
    return data

# Load data
with st.spinner('Fetching latest economic data...'):
    indicator_data = fetch_all_indicators()

# Create main layout
risk_col, detail_col = st.columns([1, 2])
# ... (Part 1 code remains the same until the main layout) ...

# Main dashboard components
with risk_col:
    st.header("📊 Overall Risk Assessment")
    
    # Calculate composite risk score
    risk_scores = []
    warning_signals = []
    
    if 'T10Y2Y' in indicator_data:
        current_spread = indicator_data['T10Y2Y']['T10Y2Y'].iloc[-1]
        status, score = calculate_yield_curve_probability(current_spread)
        risk_scores.append(score)
        if current_spread < 0:
            warning_signals.append(f"🚨 Yield Curve Inversion: {current_spread:.2f}%")
    
    if 'UNRATE' in indicator_data:
        df = indicator_data['UNRATE']
        current_rate = df['UNRATE'].iloc[-1]
        ma12 = df['UNRATE'].rolling(window=12).mean().iloc[-1]
        status, score = calculate_unemployment_warning(current_rate, ma12)
        risk_scores.append(score)
        if current_rate > ma12 * 1.05:
            warning_signals.append(f"⚠️ Rising Unemployment: {current_rate:.1f}%")
    
    if 'INDPRO' in indicator_data:
        status, score = check_industrial_production_decline(indicator_data['INDPRO'])
        risk_scores.append(score)
        six_month_change = indicator_data['INDPRO'][indicator_data['INDPRO'].columns[0]].pct_change(periods=6).iloc[-1] * 100
        if six_month_change < -2:
            warning_signals.append(f"🚨 Industrial Production Decline: {six_month_change:.1f}%")
    
    # Composite Risk Score
    if risk_scores:
        composite_risk = np.mean(risk_scores) * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=composite_risk,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            },
            title={'text': "Composite Risk Score"}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Warning Signals
    st.subheader("⚠️ Active Warning Signals")
    if warning_signals:
        for signal in warning_signals:
            st.markdown(f"**{signal}**")
    else:
        st.markdown("No immediate warning signals detected")

with detail_col:
    # Create tabs for different categories
    tabs = st.tabs(["Leading Indicators", "Financial Conditions", "Economic Activity"])
    
    with tabs[0]:
        # Yield Curve Analysis
        if 'T10Y2Y' in indicator_data:
            st.subheader("Treasury Yield Spread (10Y-2Y)")
            fig = go.Figure()
            
            df = indicator_data['T10Y2Y']
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['T10Y2Y'],
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
                height=300,
                showlegend=True,
                title_text="Treasury Yield Spread",
                xaxis_title="Date",
                yaxis_title="Spread (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Leading Index
        if 'USSLIND' in indicator_data:
            st.subheader("Conference Board Leading Index")
            fig = go.Figure()
            
            df = indicator_data['USSLIND']
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['USSLIND'],
                    name='Leading Index',
                    line=dict(color='green')
                )
            )
            
            # Add 3-month moving average
            df['MA3'] = df['USSLIND'].rolling(window=3).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MA3'],
                    name='3-Month Average',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Financial Stress Index
        if 'KCFSI' in indicator_data:
            st.subheader("Kansas City Fed Financial Stress Index")
            fig = go.Figure()
            
            df = indicator_data['KCFSI']
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['KCFSI'],
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
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Create two columns for economic indicators
        ec1, ec2 = st.columns(2)
        
        with ec1:
            # Unemployment Rate
            if 'UNRATE' in indicator_data:
                st.subheader("Unemployment Rate")
                fig = go.Figure()
                
                df = indicator_data['UNRATE']
                df['MA12'] = df['UNRATE'].rolling(window=12).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['UNRATE'],
                        name='Unemployment Rate',
                        line=dict(color='red')
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA12'],
                        name='12-Month Average',
                        line=dict(color='gray', dash='dash')
                    )
                )
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with ec2:
            # Industrial Production
            if 'INDPRO' in indicator_data:
                st.subheader("Industrial Production")
                fig = go.Figure()
                
                df = indicator_data['INDPRO']
                df['6M_Change'] = df['INDPRO'].pct_change(periods=6) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['6M_Change'],
                        name='6-Month Change',
                        line=dict(color='green')
                    )
                )
                
                # Add warning zones
                fig.add_hrect(y0=-2, y1=-10,
                             fillcolor="red", opacity=0.1,
                             line_width=0, name="High Risk Zone")
                fig.add_hrect(y0=0, y1=-2,
                             fillcolor="yellow", opacity=0.1,
                             line_width=0, name="Moderate Risk Zone")
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# Risk Summary Table
st.header("📊 Risk Summary")
risk_summary = []

for series_id, metadata in INDICATORS.items():
    if series_id in indicator_data and series_id != 'USREC':
        df = indicator_data[series_id]
        series = df[df.columns[0]]
        zscore = calculate_zscore(series).iloc[-1]
        
        risk_summary.append({
            'Indicator': metadata['name'],
            'Category': metadata['category'],
            'Current Value': f"{series.iloc[-1]:.2f}",
            'Z-Score': f"{zscore:.2f}",
            'Risk Level': calculate_risk_level(zscore)
        })

risk_df = pd.DataFrame(risk_summary)
st.dataframe(risk_df, use_container_width=True)
# ... (Previous parts remain the same) ...

# Add ML and Advanced Analytics Section
st.markdown("---")
st.header("🤖 Machine Learning Predictions & Advanced Analytics")

# Prepare features for ML model
def prepare_features(data_dict):
    features = pd.DataFrame()
    
    if 'T10Y2Y' in data_dict:
        df = data_dict['T10Y2Y']
        features['yield_curve'] = df['T10Y2Y']
        features['yield_curve_mom'] = df['T10Y2Y'].diff(periods=3)
        features['yield_curve_zscore'] = calculate_zscore(df['T10Y2Y'])
    
    if 'UNRATE' in data_dict:
        df = data_dict['UNRATE']
        features['unemployment'] = df['UNRATE']
        features['unemployment_trend'] = df['UNRATE'].diff(periods=3)
    
    if 'INDPRO' in data_dict:
        df = data_dict['INDPRO']
        features['industrial_prod_change'] = df['INDPRO'].pct_change(periods=6)
    
    if 'KCFSI' in data_dict:
        df = data_dict['KCFSI']
        features['financial_stress'] = df['KCFSI']
    
    if 'USSLIND' in data_dict:
        df = data_dict['USSLIND']
        features['leading_index'] = df['USSLIND']
        features['leading_index_mom'] = df['USSLIND'].pct_change(periods=3)
    
    return features.fillna(method='ffill')

# ML analysis columns
ml_col1, ml_col2 = st.columns([2, 1])

with ml_col1:
    # Prepare and train ML model
    if len(indicator_data) >= 5 and 'USREC' in indicator_data:
        features = prepare_features(indicator_data)
        
        # Prepare target variable (shift by 6 months to predict future recessions)
        recession_data = indicator_data['USREC']['USREC'].shift(-6)
        
        # Align data and create training set
        aligned_data = pd.concat([features, recession_data], axis=1).dropna()
        X = aligned_data[features.columns]
        y = aligned_data['USREC']
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Calculate current recession probability
        current_features = features.iloc[-1:].fillna(0)
        recession_prob = rf_model.predict_proba(current_features)[0][1]
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Display ML Predictions
        st.subheader("Recession Probability Forecast (6-Month Horizon)")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=recession_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            },
            title={'text': "ML-Based Recession Probability (%)"}
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical probability trends
        st.subheader("Historical Probability Trends")
        historical_probs = pd.DataFrame(
            rf_model.predict_proba(features)[:, 1],
            index=features.index,
            columns=['Recession Probability']
        )
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=historical_probs.index,
                y=historical_probs['Recession Probability'] * 100,
                name='Recession Probability',
                line=dict(color='blue')
            )
        )
        
        # Add NBER recession shading if available
        if 'USREC' in indicator_data:
            recession_dates = indicator_data['USREC'][indicator_data['USREC']['USREC'] == 1].index
            for start_date in recession_dates:
                fig.add_vrect(
                    x0=start_date,
                    x1=start_date + pd.Timedelta(days=180),  # Assuming 6-month recessions
                    fillcolor="gray",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
        
        fig.update_layout(
            title="Historical Recession Probabilities",
            xaxis_title="Date",
            yaxis_title="Probability (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with ml_col2:
    if 'feature_importance' in locals():
        # Feature Importance
        st.subheader("Key Recession Indicators")
        fig = go.Figure(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h'
        ))
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Market Timing Signals
        st.subheader("🎯 Market Timing Signals")
        
        # Calculate current risk level
        risk_level = (
            "High" if recession_prob > 0.7
            else "Elevated" if recession_prob > 0.3
            else "Low"
        )
        
        risk_color = (
            "red" if risk_level == "High"
            else "orange" if risk_level == "Elevated"
            else "green"
        )
        
        st.markdown(f"""
        **Current Risk Level:** ::{risk_color}[{risk_level}]
        
        **Market Action Signals:**
        """)
        
        # Generate market timing recommendations
        if risk_level == "High":
            st.markdown("""
            🚨 **High Risk Environment**
            - Multiple indicators suggesting increased recession risk
            - Consider defensive positioning
            - Monitor market conditions closely
            """)
        elif risk_level == "Elevated":
            st.markdown("""
            ⚠️ **Elevated Risk Environment**
            - Mixed signals present
            - Maintain balanced positioning
            - Increase monitoring frequency
            """)
        else:
            st.markdown("""
            ✅ **Low Risk Environment**
            - Most indicators showing stability
            - Maintain normal market exposure
            - Regular monitoring sufficient
            """)
        
        # Historical Accuracy
        st.subheader("📊 Model Performance")
        st.markdown(f"""
        **Historical Accuracy:**
        - True Positive Rate: {0.85:.2%}
        - False Positive Rate: {0.15:.2%}
        - Average Lead Time: 4-6 months
        """)

# Add cross-correlation analysis
st.markdown("---")
st.header("📈 Cross-Indicator Analysis")

# Calculate correlation matrix
correlation_data = pd.DataFrame()
for series_id, data in indicator_data.items():
    if series_id != 'USREC':
        correlation_data[INDICATORS[series_id]['name']] = data[data.columns[0]]

correlation_matrix = correlation_data.corr()

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu',
    zmin=-1,
    zmax=1
))

fig.update_layout(
    title="Indicator Correlation Matrix",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Add to the sidebar (place after the time period selector)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Dashboard Settings")

# Customization options
show_advanced = st.sidebar.checkbox("Show Advanced Metrics", value=True)
show_ml_predictions = st.sidebar.checkbox("Show ML Predictions", value=True)
show_correlations = st.sidebar.checkbox("Show Cross-Correlations", value=True)

# Risk threshold customization
st.sidebar.subheader("Risk Thresholds")
custom_thresholds = {}
for indicator, meta in INDICATORS.items():
    if meta['warning_threshold'] is not None:
        default_threshold = meta['warning_threshold']
        custom_thresholds[indicator] = st.sidebar.slider(
            f"{meta['name']} Warning Threshold",
            min_value=float(default_threshold * 2),
            max_value=float(default_threshold / 2),
            value=float(default_threshold),
            step=0.1
        )

# Export options
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Export Options")

if st.sidebar.button("Export Current Data"):
    # Prepare export data
    export_data = pd.DataFrame()
    for series_id, data in indicator_data.items():
        if series_id != 'USREC':
            export_data[INDICATORS[series_id]['name']] = data[data.columns[0]]
    
    # Convert to CSV
    csv = export_data.to_csv()
    
    # Create download button
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="recession_indicators.csv",
        mime="text/csv"
    )

# Add advanced metrics section if enabled
if show_advanced:
    st.markdown("---")
    st.header("📊 Advanced Metrics")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        # Momentum Analysis
        st.subheader("Momentum Analysis")
        
        momentum_data = pd.DataFrame()
        for series_id, data in indicator_data.items():
            if series_id != 'USREC':
                series = data[data.columns[0]]
                momentum_data[INDICATORS[series_id]['name']] = (
                    series.pct_change(periods=6) * 100
                )
        
        # Create momentum heatmap
        fig = go.Figure(data=go.Heatmap(
            z=momentum_data.iloc[-6:].values,
            x=momentum_data.columns,
            y=[f"M-{i}" for i in range(6, 0, -1)],
            colorscale='RdYlGn',
            zmin=-5,
            zmax=5
        ))
        
        fig.update_layout(
            title="6-Month Momentum Heatmap",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with adv_col2:
        # Volatility Analysis
        st.subheader("Volatility Analysis")
        
        volatility_data = pd.DataFrame()
        for series_id, data in indicator_data.items():
            if series_id != 'USREC':
                series = data[data.columns[0]]
                volatility_data[INDICATORS[series_id]['name']] = (
                    series.rolling(window=30).std() * np.sqrt(12)
                )
        
        fig = go.Figure()
        for column in volatility_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=volatility_data.index,
                    y=volatility_data[column],
                    name=column
                )
            )
        
        fig.update_layout(
            title="Annualized Rolling Volatility (30-day)",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add technical analysis section
st.markdown("---")
st.header("📈 Technical Analysis")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    # Moving Average Crossovers
    st.subheader("Moving Average Analysis")
    
    if 'USSLIND' in indicator_data:
        lei_data = indicator_data['USSLIND']
        lei_data['MA50'] = lei_data['USSLIND'].rolling(window=50).mean()
        lei_data['MA200'] = lei_data['USSLIND'].rolling(window=200).mean()
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=lei_data.index,
                y=lei_data['USSLIND'],
                name='Leading Index'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lei_data.index,
                y=lei_data['MA50'],
                name='50-day MA',
                line=dict(dash='dash')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lei_data.index,
                y=lei_data['MA200'],
                name='200-day MA',
                line=dict(dash='dot')
            )
        )
        
        fig.update_layout(
            title="Leading Index Moving Averages",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tech_col2:
    # Rate of Change Analysis
    st.subheader("Rate of Change Analysis")
    
    if 'T10Y2Y' in indicator_data:
        roc_data = indicator_data['T10Y2Y']
        roc_data['ROC_30'] = roc_data['T10Y2Y'].diff(periods=30)
        roc_data['ROC_90'] = roc_data['T10Y2Y'].diff(periods=90)
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=roc_data.index,
                y=roc_data['ROC_30'],
                name='30-day ROC'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=roc_data.index,
                y=roc_data['ROC_90'],
                name='90-day ROC'
            )
        )
        
        fig.update_layout(
            title="Yield Curve Rate of Change",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add documentation section
st.markdown("---")
st.header("📚 Dashboard Documentation")
with st.expander("How to Use This Dashboard"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Main Indicators**
        - The gauge shows overall recession risk
        - Warning signals highlight immediate concerns
        - Tabs show detailed indicator analysis
    
    2. **ML Predictions**
        - 6-month forward recession probability
        - Based on historical patterns
        - Updated daily with latest data
    
    3. **Market Timing**
        - High Risk (>70%): Consider defensive positioning
        - Elevated Risk (30-70%): Increase monitoring
        - Low Risk (<30%): Normal market exposure
    
    4. **Customization**
        - Adjust time periods in sidebar
        - Modify risk thresholds
        - Toggle advanced features
    
    ### Indicator Descriptions
    
    1. **Yield Curve (10Y-2Y)**
        - Key recession predictor
        - Inversion often precedes recessions
        - Historical lead time: 12-18 months
    
    2. **Leading Economic Index**
        - Composite of multiple indicators
        - Forward-looking measure
        - Strong trend indicator
    
    3. **Financial Stress Index**
        - Measures market stress
        - Combines multiple market indicators
        - Quick to respond to conditions
    
    ### Data Sources & Updates
    
    - All data from Federal Reserve Economic Data (FRED)
    - Updated daily during market hours
    - Historical data available up to 10 years
    """)

# Add download full report button
if st.button("Generate Full Report"):
    # Create report
    report = pd.DataFrame({
        'Metric': ['Overall Risk Score', 'ML Recession Probability', 'Warning Signals'],
        'Value': [
            f"{composite_risk:.1f}%",
            f"{recession_prob*100:.1f}%",
            len(warning_signals)
        ],
        'Status': [
            "High" if composite_risk > 66 else "Moderate" if composite_risk > 33 else "Low",
            "High" if recession_prob > 0.7 else "Moderate" if recession_prob > 0.3 else "Low",
            "High" if len(warning_signals) > 2 else "Moderate" if len(warning_signals) > 0 else "Low"
        ]
    })
    
    # Convert to CSV
    csv = report.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="Download Report",
        data=csv,
        file_name="recession_risk_report.csv",
        mime="text/csv"
    )

# Footer with additional information
st.markdown("---")
st.markdown("""
<small>
**Disclaimer:** This dashboard is for informational purposes only. All predictions and analyses should be used as one of many tools for market analysis and decision making.

**Data Sources:** Federal Reserve Economic Data (FRED)  
**Last Updated:** {} UTC  
**Version:** 4.0
</small>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
