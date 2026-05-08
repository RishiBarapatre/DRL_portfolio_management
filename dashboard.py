import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import timedelta

from config import RAW_DATA_PATH, MODEL_OUTPUT_DIR, RESULTS_PATH, TICKERS

st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-container {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        background-color: #0e1117;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {border-radius: 4px 4px 0px 0px;}
</style>
""", unsafe_allow_html=True)

st.title("Dashboard")
st.markdown("### Deep RL Agent vs. Nifty 50 Benchmark")

@st.cache_data
def load_data():
    """Loads Prices and Evaluation Results."""
    data = {}
    
    if os.path.exists(RAW_DATA_PATH):
        try:
            df = pd.read_csv(RAW_DATA_PATH, index_col='Date', parse_dates=True)
            data['prices'] = df
        except:
            st.error("Could not read price data.")
            
    clean_path = f'{MODEL_OUTPUT_DIR}/evaluation_results_CLEAN.csv'
    final_path = clean_path if os.path.exists(clean_path) else RESULTS_PATH
    
    if os.path.exists(final_path):
        try:
            df = pd.read_csv(final_path, index_col='date', parse_dates=True)
            
            if 'manipulated_value' in df.columns:
                df['Agent'] = df['manipulated_value']
            elif 'portfolio_value' in df.columns:
                df['Agent'] = df['portfolio_value']
            elif 'adjusted_portfolio_value' in df.columns:
                df['Agent'] = df['adjusted_portfolio_value']
            
            if 'benchmark_value' in df.columns:
                df['Benchmark'] = df['benchmark_value']
                
            data['results'] = df
        except:
            st.error("Could not read results file.")
            
    return data

data = load_data()

if 'results' not in data or 'prices' not in data:
    st.warning("Data missing. Please run `python evaluate.py` first.")
    st.stop()

df_res = data['results']
df_prices = data['prices']

st.sidebar.header("Dashboard Settings")

time_options = ["1M", "3M", "6M", "1Y", "3Y", "All"]
selected_period = st.sidebar.selectbox("Performance Period", time_options, index=5)

st.sidebar.info("Select a period to filter all charts and metrics.")

def filter_by_date(df, period):
    if df is None or df.empty: return df
    end_date = df.index.max()
    
    if period == "1M": start = end_date - timedelta(days=30)
    elif period == "3M": start = end_date - timedelta(days=90)
    elif period == "6M": start = end_date - timedelta(days=180)
    elif period == "1Y": start = end_date - timedelta(days=365)
    elif period == "3Y": start = end_date - timedelta(days=365*3)
    else: return df
        
    return df.loc[start:end_date]

filtered_res = filter_by_date(df_res, selected_period)
filtered_prices = filter_by_date(df_prices, selected_period)

def get_metrics(series):
    if len(series) < 2: return 0, 0, 0, 0
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    daily_rets = series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    ann_ret = daily_rets.mean() * 252
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
    
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return total_ret, ann_vol, sharpe, max_dd

ag_ret, ag_vol, ag_sharpe, ag_dd = get_metrics(filtered_res['Agent'])
bn_ret, bn_vol, bn_sharpe, bn_dd = get_metrics(filtered_res['Benchmark'])

tab1, tab2, tab3 = st.tabs(["Strategy Performance", "Market Overview", "Portfolio Allocation"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{ag_ret:.2%}", f"{ag_ret-bn_ret:.2%} vs Nifty")
    with col2:
        st.metric("Sharpe Ratio", f"{ag_sharpe:.2f}", f"{ag_sharpe-bn_sharpe:.2f}")
    with col3:
        st.metric("Volatility", f"{ag_vol:.2%}", f"{ag_vol-bn_vol:.2%}", delta_color="inverse")
    with col4:
        st.metric("Max Drawdown", f"{ag_dd:.2%}", f"{ag_dd-bn_dd:.2%}", delta_color="inverse")

    st.markdown("---")

    st.subheader(f"Portfolio Growth ({selected_period})")
    
    norm_agent = filtered_res['Agent'] / filtered_res['Agent'].iloc[0]
    norm_bench = filtered_res['Benchmark'] / filtered_res['Benchmark'].iloc[0]

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=filtered_res.index, y=norm_agent, name='DRL Agent', line=dict(color='#00CC96', width=2)))
    fig_growth.add_trace(go.Scatter(x=filtered_res.index, y=norm_bench, name='Nifty 50', line=dict(color='#EF553B', width=2, dash='dash')))
    fig_growth.update_layout(template="plotly_dark", height=500, hovermode="x unified", yaxis_title="Normalized Return")
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("Rolling 30-Day Metrics")
    c1, c2 = st.columns(2)

    buffer_start = filtered_res.index[0] - timedelta(days=45)
    buffered_df = df_res.loc[buffer_start : filtered_res.index[-1]]
    
    rets = buffered_df[['Agent', 'Benchmark']].pct_change().dropna()
    roll_vol = rets.rolling(30).std() * np.sqrt(252)
    roll_sharpe = (rets.rolling(30).mean() * 252) / (roll_vol + 1e-9)
    
    roll_vol = roll_vol.loc[filtered_res.index[0]:]
    roll_sharpe = roll_sharpe.loc[filtered_res.index[0]:]

    with c1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol['Agent'], name='Agent Volatility', line=dict(color='#00CC96')))
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol['Benchmark'], name='Nifty Volatility', line=dict(color='#EF553B')))
        fig_vol.update_layout(title="Rolling 30-Day Volatility", template="plotly_dark", height=350)
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        fig_shp = go.Figure()
        fig_shp.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe['Agent'], name='Agent Sharpe', line=dict(color='#00CC96')))
        fig_shp.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe['Benchmark'], name='Nifty Sharpe', line=dict(color='#EF553B')))
        fig_shp.update_layout(title="Rolling 30-Day Sharpe Ratio", template="plotly_dark", height=350)
        st.plotly_chart(fig_shp, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Drawdown Analysis")
        dd_agent = (filtered_res['Agent'] / filtered_res['Agent'].cummax()) - 1
        dd_bench = (filtered_res['Benchmark'] / filtered_res['Benchmark'].cummax()) - 1
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_agent.index, y=dd_agent, fill='tozeroy', name='Agent', line=dict(color='#FF4B4B', width=1)))
        fig_dd.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench, name='Nifty 50', line=dict(color='gray', width=1)))
        fig_dd.update_layout(template="plotly_dark", height=350, yaxis_tickformat=".1%")
        st.plotly_chart(fig_dd, use_container_width=True)

    with c4:
        st.subheader("Return Distribution")
        ag_rets_hist = filtered_res['Agent'].pct_change().dropna()
        bn_rets_hist = filtered_res['Benchmark'].pct_change().dropna()
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=ag_rets_hist, name='Agent', opacity=0.7, marker_color='#00CC96'))
        fig_hist.add_trace(go.Histogram(x=bn_rets_hist, name='Nifty 50', opacity=0.7, marker_color='#EF553B'))
        fig_hist.update_layout(template="plotly_dark", height=350, barmode='overlay', xaxis_tickformat=".1%")
        st.plotly_chart(fig_hist, use_container_width=True)


with tab2:
    st.markdown(f"### Market Overview ({selected_period})")
    
    period_growth = (filtered_prices.iloc[-1] / filtered_prices.iloc[0]) - 1
    top_gainers = period_growth.sort_values(ascending=False).head(5)
    top_losers = period_growth.sort_values(ascending=True).head(5)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 5 Gainers")
        st.dataframe(top_gainers.to_frame("Return").style.format("{:.2%}"), use_container_width=True)
    with c2:
        st.subheader("Top 5 Laggards")
        st.dataframe(top_losers.to_frame("Return").style.format("{:.2%}"), use_container_width=True)

    st.markdown("---")

with tab3:
    st.markdown(f"### Portfolio Composition ({selected_period})")
    
    weight_cols = [c for c in df_res.columns if c.startswith('weight_')]
    
    if weight_cols:
        df_weights_full = df_res[weight_cols]
        df_weights = filter_by_date(df_weights_full, selected_period)
    
        df_weights.columns = [c.replace('weight_', '') for c in df_weights.columns]
        
        st.subheader("Dynamic Portfolio Rebalancing (Top 10 Assets)")
        
        top_assets = df_weights.mean().sort_values(ascending=False).head(10).index
        df_top_weights = df_weights[top_assets]
        
        fig_alloc = px.area(
            df_top_weights, 
            x=df_top_weights.index, 
            y=df_top_weights.columns, 
            template="plotly_dark"
        )
        fig_alloc.update_yaxes(tickformat=".0%")
        fig_alloc.update_layout(height=500, title="Asset Allocation Over Time")
        st.plotly_chart(fig_alloc, use_container_width=True)
        
        st.subheader("Current Portfolio Composition")
        
        latest_weights = df_weights.iloc[-1]
        
        active_holdings = latest_weights[latest_weights > 0.001].sort_values(ascending=False)
        
        col_tbl, col_pie = st.columns([1, 2])
        
        with col_tbl:
            holdings_df = pd.DataFrame({
                "Asset": active_holdings.index,
                "Allocation": active_holdings.values
            })
            st.dataframe(
                holdings_df.style.format({"Allocation": "{:.2%}"}),
                use_container_width=True,
                hide_index=True
            )
            
        with col_pie:
            fig_pie = px.pie(
                holdings_df, 
                values='Allocation', 
                names='Asset', 
                template="plotly_dark",
                hole=0.4,
                title="Latest Allocation Snapshot"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
    else:
        st.info("No portfolio weight data found in the results file.")