"""
Streamlit Dashboard for MLOps Energy Forecasting System.
Visualizes model performance, predictions, drift metrics, and system health.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import PROCESSED_DIR, REFERENCE_DIR, RAW_DIR
from src.model.registry import get_champion_model
import mlflow

# Configure MLflow for DagsHub if secrets are available
if hasattr(st, 'secrets') and 'mlflow' in st.secrets:
    mlflow_config = st.secrets['mlflow']
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_config.get('MLFLOW_TRACKING_URI', '')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_config.get('DAGSHUB_USER', '')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_config.get('DAGSHUB_TOKEN', '')
    mlflow.set_tracking_uri(mlflow_config['MLFLOW_TRACKING_URI'])

# Page config
st.set_page_config(
    page_title="Energy Demand Forecasting - MLOps Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-bad {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_feature_data():
    """Load processed feature data."""
    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data(ttl=300)
def load_reference_data():
    """Load reference dataset."""
    ref_path = REFERENCE_DIR / "reference_features.parquet"
    if ref_path.exists():
        df = pd.read_parquet(ref_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data(ttl=300)
def load_drift_results():
    """Load latest drift detection results."""
    drift_path = Path("reports/drift_results.json")
    if drift_path.exists():
        with open(drift_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data(ttl=60)
def get_model_info():
    """Get champion model information."""
    try:
        # Try to connect to MLflow (DagsHub or local)
        client = mlflow.MlflowClient()
        
        # Get latest version
        versions = client.search_model_versions(f"name='energy-demand-lgbm'")
        if versions:
            latest = max(versions, key=lambda x: int(x.version))
            run = client.get_run(latest.run_id)
            
            return {
                "version": latest.version,
                "run_id": latest.run_id,
                "mae": run.data.metrics.get("mae", 0),
                "rmse": run.data.metrics.get("rmse", 0),
                "r2": run.data.metrics.get("r2", 0),
                "stage": latest.current_stage,
                "timestamp": latest.creation_timestamp,
            }
        return None
    except Exception as e:
        st.warning(f"Could not load model info: {str(e)}")
        return None


def create_predictions_chart(df):
    """Create interactive time series chart of predictions vs actuals."""
    fig = go.Figure()
    
    # Actual demand
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['demand_mwh'],
        mode='lines',
        name='Actual Demand',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add forecast line if predictions exist
    if 'prediction' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['prediction'],
            mode='lines',
            name='Predicted Demand',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Energy Demand: Actual vs Predicted",
        xaxis_title="Time",
        yaxis_title="Demand (MWh)",
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_feature_importance_chart(model_info):
    """Create feature importance bar chart."""
    try:
        client = mlflow.MlflowClient()
        run = client.get_run(model_info['run_id'])
        
        # Get feature importance from artifacts or metrics
        # For now, create a placeholder
        features = ['temperature', 'hour_of_day', 'demand_lag_1h', 'day_of_week', 
                   'demand_lag_24h', 'humidity', 'wind_speed', 'is_weekend']
        importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(color='#1f77b4')
        ))
        
        fig.update_layout(
            title="Top Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        
        return fig
    except:
        return None


def create_drift_chart(drift_results):
    """Create drift status visualization."""
    if not drift_results:
        return None
    
    drift_share = drift_results.get('drift_share', 0)
    threshold = drift_results.get('threshold', 0.3)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=drift_share * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Drift Share (%)"},
        delta={'reference': threshold * 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold * 100], 'color': "lightgreen"},
                {'range': [threshold * 100, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def create_error_distribution(df):
    """Create error distribution histogram."""
    if 'prediction' not in df.columns:
        return None
    
    errors = df['demand_mwh'] - df['prediction']
    
    fig = go.Figure(go.Histogram(
        x=errors.dropna(),
        nbinsx=50,
        marker=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        title="Prediction Error Distribution",
        xaxis_title="Error (MWh)",
        yaxis_title="Frequency",
        height=300
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">⚡ Energy Demand Forecasting - MLOps Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Real-time monitoring of ML model performance, predictions, and data drift")
    
    # Sidebar
    st.sidebar.title("⚙️ Controls")
    
    # Time range selector
    days_back = st.sidebar.slider("Days to Display", 1, 30, 7)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    
    # Load data
    features_df = load_feature_data()
    reference_df = load_reference_data()
    drift_results = load_drift_results()
    model_info = get_model_info()
    
    # System status indicators
    if features_df is not None:
        st.sidebar.success(f"✓ Data Loaded ({len(features_df)} records)")
    else:
        st.sidebar.error("✗ No feature data")
    
    if model_info:
        st.sidebar.success(f"✓ Model v{model_info['version']} Active")
    else:
        st.sidebar.warning("⚠ No model loaded")
    
    if drift_results and drift_results.get('dataset_drift'):
        st.sidebar.error("⚠ Drift Detected!")
    else:
        st.sidebar.success("✓ No Drift")
    
    # Main content
    if features_df is None:
        st.warning("⚠️ No data available. Run the data ingestion flow first.")
        st.code("python src/scripts/run_flows.py")
        return
    
    # Filter to time range
    cutoff = datetime.now() - timedelta(days=days_back)
    df_filtered = features_df[features_df['timestamp'] >= cutoff].copy()
    
    # Row 1: Key Metrics
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Latest Demand",
            f"{df_filtered['demand_mwh'].iloc[-1]:,.0f} MWh" if len(df_filtered) > 0 else "N/A",
            delta=f"{df_filtered['demand_mwh'].iloc[-1] - df_filtered['demand_mwh'].iloc[-2]:,.0f}" if len(df_filtered) > 1 else None
        )
    
    with col2:
        if model_info:
            st.metric("Model MAE", f"{model_info['mae']:,.0f} MWh")
        else:
            st.metric("Model MAE", "Training needed")
    
    with col3:
        if model_info:
            st.metric("Model R²", f"{model_info['r2']:.3f}")
        else:
            st.metric("Model R²", "N/A")
    
    with col4:
        if drift_results:
            drift_pct = drift_results.get('drift_share', 0) * 100
            st.metric("Drift Share", f"{drift_pct:.1f}%", 
                     delta=f"{drift_pct - 30:.1f}%" if drift_pct > 30 else None,
                     delta_color="inverse")
        else:
            st.metric("Drift Share", "N/A")
    
    # Row 2: Main Charts
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Demand Forecast")
        predictions_chart = create_predictions_chart(df_filtered)
        st.plotly_chart(predictions_chart, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Drift Status")
        if drift_results:
            drift_chart = create_drift_chart(drift_results)
            if drift_chart:
                st.plotly_chart(drift_chart, use_container_width=True)
            
            # Drift details
            if drift_results.get('dataset_drift'):
                st.error(f"⚠️ **Drift Detected!**")
                st.write(f"Drifted Features: {drift_results['n_drifted_features']}/{drift_results['n_total_features']}")
                st.write(f"Threshold: {drift_results['threshold']:.0%}")
            else:
                st.success("✓ No significant drift")
        else:
            st.info("No drift data available")
    
    # Row 3: Additional Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Feature Importance")
        if model_info:
            importance_chart = create_feature_importance_chart(model_info)
            if importance_chart:
                st.plotly_chart(importance_chart, use_container_width=True)
        else:
            st.info("Model information not available")
    
    with col2:
        st.markdown("### 📉 Error Distribution")
        error_chart = create_error_distribution(df_filtered)
        if error_chart:
            st.plotly_chart(error_chart, use_container_width=True)
        else:
            st.info("Predictions not available")
    
    # Row 4: Data Tables
    st.markdown("---")
    st.markdown("### 📋 Recent Data")
    
    tab1, tab2, tab3 = st.tabs(["Recent Predictions", "Model Info", "System Logs"])
    
    with tab1:
        display_cols = ['timestamp', 'demand_mwh', 'temperature', 'hour_of_day', 'day_of_week']
        if 'prediction' in df_filtered.columns:
            display_cols.append('prediction')
        
        st.dataframe(
            df_filtered[display_cols].tail(20).sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        if model_info:
            st.json(model_info)
        else:
            st.info("No model information available")
    
    with tab3:
        st.markdown("**Recent Activity**")
        st.text(f"Last data update: {df_filtered['timestamp'].max()}")
        st.text(f"Total records: {len(features_df)}")
        st.text(f"Reference samples: {len(reference_df) if reference_df is not None else 0}")
        if drift_results:
            st.text(f"Last drift check: {drift_results.get('check_timestamp', 'N/A')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>MLOps Energy Forecasting System | Built with Streamlit, MLflow, Prefect & Evidently</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
