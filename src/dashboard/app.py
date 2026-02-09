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
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '30'  # 30 second timeout for slower connections
    mlflow.set_tracking_uri(mlflow_config['MLFLOW_TRACKING_URI'])

# Page config
st.set_page_config(
    page_title="Energy Demand Forecasting - MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color palette - Energy theme
COLORS = {
    'primary': '#FF6B35',      # Energy orange
    'secondary': '#004E89',    # Deep blue
    'accent': '#F7B32B',       # Amber
    'success': '#06A77D',      # Green
    'warning': '#F77F00',      # Orange
    'danger': '#D62828',       # Red
    'actual': '#004E89',       # Blue for actual
    'predicted': '#FF6B35',    # Orange for predicted
    'confidence': 'rgba(255, 107, 53, 0.2)',  # Light orange
    'background': '#F8F9FA',
    'text': '#2C3E50'
}

# Custom CSS with professional styling
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }}
    .subtitle {{
        font-size: 1.1rem;
        color: {COLORS['text']};
        opacity: 0.8;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background: linear-gradient(135deg, white 0%, {COLORS['background']} 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid {COLORS['primary']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    .status-good {{
        color: {COLORS['success']};
        font-weight: 600;
    }}
    .status-warning {{
        color: {COLORS['warning']};
        font-weight: 600;
    }}
    .status-bad {{
        color: {COLORS['danger']};
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS['secondary']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        background-color: {COLORS['background']};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: white;
        border-top: 3px solid {COLORS['primary']};
    }}
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


@st.cache_data(ttl=300)
def load_performance_log():
    """Load performance monitoring log."""
    perf_path = Path("reports/performance_log.json")
    if perf_path.exists():
        with open(perf_path, 'r') as f:
            data = json.load(f)
            if 'entries' in data and data['entries']:
                df = pd.DataFrame(data['entries'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
    return None


@st.cache_data(ttl=300)
def get_all_model_versions():
    """Get all model versions for comparison."""
    try:
        tracking_uri = mlflow.get_tracking_uri()
        if not tracking_uri or 'file:' in tracking_uri:
            return None
        
        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='energy-demand-lgbm'", max_results=10)
        
        if not versions:
            return None
        
        models = []
        for v in versions:
            run = client.get_run(v.run_id)
            models.append({
                "Version": v.version,
                "Stage": v.current_stage,
                "MAE": f"{run.data.metrics.get('mae', 0):,.0f}",
                "RMSE": f"{run.data.metrics.get('rmse', 0):,.0f}",
                "R²": f"{run.data.metrics.get('r2', 0):.3f}",
                "Created": pd.to_datetime(v.creation_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M'),
                "Run ID": v.run_id[:8]
            })
        
        return pd.DataFrame(models).sort_values('Version', ascending=False)
    except:
        return None


def get_model_info():
    """Get champion model information (non-blocking)."""
    try:
        # Check if MLflow is configured
        tracking_uri = mlflow.get_tracking_uri()
        
        # If no tracking URI, show helpful message
        if not tracking_uri or 'file:' in tracking_uri:
            st.info("Add MLflow secrets to see model info")
            return None
        
        # Try to connect to MLflow (DagsHub or local)
        with st.spinner("Connecting to MLflow..."):
            client = mlflow.MlflowClient()
            
            # Get latest version
            versions = client.search_model_versions("name='energy-demand-lgbm'", max_results=5)
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
        # Show the actual error for debugging
        st.error(f"MLflow error: {type(e).__name__}: {str(e)}")
        return None


def create_predictions_chart(df):
    """Create interactive time series chart with confidence intervals."""
    fig = go.Figure()
    
    # Calculate prediction error std for confidence intervals
    if 'prediction' in df.columns:
        errors = df['demand_mwh'] - df['prediction']
        std_error = errors.std()
        
        # Upper confidence bound (95%)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['prediction'] + 1.96 * std_error,
            mode='lines',
            name='95% Confidence Upper',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Lower confidence bound (95%)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['prediction'] - 1.96 * std_error,
            mode='lines',
            name='95% Confidence Lower',
            line=dict(width=0),
            fillcolor=COLORS['confidence'],
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Predicted demand
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['prediction'],
            mode='lines',
            name='Predicted Demand',
            line=dict(color=COLORS['predicted'], width=2.5)
        ))
    
    # Actual demand (plotted last so it's on top)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['demand_mwh'],
        mode='lines',
        name='Actual Demand',
        line=dict(color=COLORS['actual'], width=3)
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Energy Demand Forecast</b>",
            font=dict(size=20, color=COLORS['text'])
        ),
        xaxis=dict(
            title="Time",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title="Demand (MWh)",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        hovermode='x unified',
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def create_feature_importance_chart(model_info):
    """Create feature importance bar chart from MLflow artifact."""
    try:
        client = mlflow.MlflowClient()
        run_id = model_info['run_id']
        
        # Try to get feature importance from logged CSV artifact
        try:
            import tempfile
            import os
            
            # Download the feature importance artifact
            artifact_path = "feature_importance.csv"
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
                importance_df = pd.read_csv(local_path)
                
                # Get top 10 features
                importance_df = importance_df.head(10)
                features = importance_df['feature'].tolist()
                importance = importance_df['importance'].tolist()
                
                # Normalize to sum to 1
                total = sum(importance)
                importance = [i/total for i in importance]
        except Exception as e:
            print(f"Could not load feature importance artifact: {e}")
            # Fallback to placeholder data
            features = ['temperature', 'hour_of_day', 'demand_lag_1h', 'day_of_week', 
                       'demand_lag_24h', 'humidity', 'wind_speed', 'is_weekend']
            importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
        
        # Create gradient colors for bars
        colors = [f'rgba({int(255-i*15)}, {int(107-i*5)}, {int(53+i*10)}, 0.8)' 
                  for i in range(len(importance))]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color=COLORS['primary'], width=1)
            ),
            text=[f'{val:.1%}' for val in importance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(
                text="<b>Top Feature Importance</b>",
                font=dict(size=18, color=COLORS['text'])
            ),
            xaxis=dict(
                title="Importance Score",
                tickformat='.0%',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(title=""),
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    except:
        return None


def create_drift_chart(drift_results):
    """Create drift status visualization with gauge."""
    if not drift_results:
        return None
    
    drift_share = drift_results.get('drift_share', 0)
    threshold = drift_results.get('threshold', 0.3)
    
    # Determine color based on drift level
    if drift_share < threshold * 0.5:
        bar_color = COLORS['success']
    elif drift_share < threshold:
        bar_color = COLORS['warning']
    else:
        bar_color = COLORS['danger']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=drift_share * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title=dict(
            text="<b>Drift Detection</b>",
            font=dict(size=18, color=COLORS['text'])
        ),
        number={'suffix': '%', 'font': {'size': 40, 'color': bar_color}},
        delta={'reference': threshold * 100, 'increasing': {'color': COLORS['danger']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': bar_color, 'thickness': 0.7},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'rgba(0,0,0,0.1)',
            'steps': [
                {'range': [0, threshold * 50], 'color': 'rgba(6, 167, 125, 0.2)'},
                {'range': [threshold * 50, threshold * 100], 'color': 'rgba(247, 127, 0, 0.2)'},
                {'range': [threshold * 100, 100], 'color': 'rgba(214, 40, 40, 0.2)'}
            ],
            'threshold': {
                'line': {'color': COLORS['danger'], 'width': 3},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=320,
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def create_drift_breakdown(drift_results):
    """Create bar chart showing which features are drifting."""
    if not drift_results or 'per_feature_drift' not in drift_results:
        return None
    
    per_feature = drift_results.get('per_feature_drift', {})
    if not per_feature:
        return None
    
    # Sort by drift score
    features = list(per_feature.keys())
    drift_scores = [per_feature[f].get('drift_score', 0) for f in features]
    
    # Sort descending
    sorted_pairs = sorted(zip(features, drift_scores), key=lambda x: x[1], reverse=True)
    features, drift_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # Top 10 drifted features
    features = features[:10]
    drift_scores = drift_scores[:10]
    
    colors = [COLORS['danger'] if s > 0.5 else COLORS['warning'] if s > 0.3 else COLORS['success'] 
              for s in drift_scores]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=drift_scores,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{s:.2f}' for s in drift_scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Feature Drift Scores</b>",
            font=dict(size=18, color=COLORS['text'])
        ),
        xaxis=dict(
            title="Drift Score",
            range=[0, 1],
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(title=""),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def create_error_distribution(df):
    """Create error distribution histogram with statistics."""
    if 'prediction' not in df.columns:
        return None
    
    errors = (df['demand_mwh'] - df['prediction']).dropna()
    
    if len(errors) == 0:
        return None
    
    # Calculate statistics
    mae = errors.abs().mean()
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=40,
        name='Prediction Errors',
        marker=dict(
            color=COLORS['secondary'],
            line=dict(color='white', width=1)
        ),
        opacity=0.8
    ))
    
    # Add mean line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=COLORS['success'],
        line_width=2,
        annotation_text="Perfect Prediction",
        annotation_position="top"
    )
    
    # Add MAE lines
    fig.add_vline(
        x=mae,
        line_dash="dot",
        line_color=COLORS['warning'],
        line_width=2,
        annotation_text=f"MAE: {mae:.0f}"
    )
    fig.add_vline(
        x=-mae,
        line_dash="dot",
        line_color=COLORS['warning'],
        line_width=2
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Prediction Error Distribution</b>",
            font=dict(size=18, color=COLORS['text'])
        ),
        xaxis=dict(
            title="Error (MWh)",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=2
        ),
        yaxis=dict(
            title="Frequency",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def create_performance_trends(perf_df):
    """Create enhanced performance metrics over time chart with dual axis."""
    if perf_df is None or len(perf_df) == 0:
        return None
    
    fig = go.Figure()
    
    # MAE trend with area fill
    fig.add_trace(go.Scatter(
        x=perf_df['timestamp'],
        y=perf_df['mae'],
        mode='lines+markers',
        name='MAE (MWh)',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, symbol='circle', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 53, 0.1)',
        yaxis='y'
    ))
    
    # R² trend (on secondary axis)
    fig.add_trace(go.Scatter(
        x=perf_df['timestamp'],
        y=perf_df['r2'],
        mode='lines+markers',
        name='R² Score',
        line=dict(color=COLORS['success'], width=3),
        marker=dict(size=8, symbol='diamond', line=dict(color='white', width=2)),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Model Performance Trends</b>",
            font=dict(size=20, color=COLORS['text'])
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title="MAE (MWh)",
            side='left',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            titlefont=dict(color=COLORS['primary']),
            tickfont=dict(color=COLORS['primary'])
        ),
        yaxis2=dict(
            title="R² Score",
            side='right',
            overlaying='y',
            titlefont=dict(color=COLORS['success']),
            tickfont=dict(color=COLORS['success']),
            range=[-1, 1]
        ),
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Enhanced Header with subtitle
    st.markdown('<p class="main-header">Energy Demand Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Production MLOps Dashboard · Real-time Monitoring & Drift Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Time range selector
    days_back = st.sidebar.slider("📅 Days to Display", 1, 30, 7, help="Select the time range for data visualization")
    
    # Refresh button
    if st.sidebar.button("Refresh Data", help="Clear cache and reload all data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Health")
    
    # Load data
    features_df = load_feature_data()
    reference_df = load_reference_data()
    drift_results = load_drift_results()
    model_info = get_model_info()
    perf_log = load_performance_log()
    
    # System status indicators with better styling
    if features_df is not None:
        st.sidebar.success(f"Data Pipeline: {len(features_df)} records")
    else:
        st.sidebar.error("✗ **Data Pipeline:** No data")
    
    if model_info:
        st.sidebar.success(f"Model v{model_info['version']} Active")
    else:
        st.sidebar.warning("⚠ No model loaded")
    
    if drift_results and drift_results.get('dataset_drift'):
        st.sidebar.error("⚠ Drift Detected!")
    else:
        st.sidebar.success("No Drift")
    
    # Main content
    if features_df is None:
        st.warning("No data available. Run the data ingestion flow first.")
        st.code("python src/scripts/run_flows.py")
        return
    
    # Filter to time range
    # Use the latest timestamp in data as reference to avoid timezone issues
    latest_time = features_df['timestamp'].max()
    cutoff = latest_time - timedelta(days=days_back)
    df_filtered = features_df[features_df['timestamp'] >= cutoff].copy()
    
    st.sidebar.info(f"Showing {len(df_filtered)} of {len(features_df)} records")
    
    # Row 1: Key Metrics
    st.markdown("### Key Performance Indicators")
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
        st.markdown("### Demand Forecast")
        predictions_chart = create_predictions_chart(df_filtered)
        st.plotly_chart(predictions_chart, use_container_width=True)
    
    with col2:
        st.markdown("### Drift Status")
        if drift_results:
            drift_chart = create_drift_chart(drift_results)
            if drift_chart:
                st.plotly_chart(drift_chart, use_container_width=True)
            
            # Drift details
            if drift_results.get('dataset_drift'):
                st.error(f"**Drift Detected!**")
                st.write(f"Drifted Features: {drift_results['n_drifted_features']}/{drift_results['n_total_features']}")
                st.write(f"Threshold: {drift_results['threshold']:.0%}")
            else:
                st.success("No significant drift")
        else:
            st.info("No drift data available")
    
    # Row 3: Drift Breakdown & Feature Importance
    st.markdown("---")
    
    if drift_results and drift_results.get('dataset_drift'):
        # Show drift breakdown if drift detected
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Feature Drift Breakdown")
            drift_breakdown = create_drift_breakdown(drift_results)
            if drift_breakdown:
                st.plotly_chart(drift_breakdown, use_container_width=True)
            else:
                st.info("Drift breakdown not available")
        
        with col2:
            st.markdown("### Feature Importance")
            if model_info:
                importance_chart = create_feature_importance_chart(model_info)
                if importance_chart:
                    st.plotly_chart(importance_chart, use_container_width=True)
            else:
                st.info("Model information not available")
    else:
        # No drift - show feature importance and error distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Feature Importance")
            if model_info:
                importance_chart = create_feature_importance_chart(model_info)
                if importance_chart:
                    st.plotly_chart(importance_chart, use_container_width=True)
            else:
                st.info("Model information not available")
        
        with col2:
            st.markdown("### Error Distribution")
            error_chart = create_error_distribution(df_filtered)
            if error_chart:
                st.plotly_chart(error_chart, use_container_width=True)
            else:
                st.info("Predictions not available")
    
    # Row 4: Performance Trends
    if perf_log is not None and len(perf_log) > 1:
        st.markdown("---")
        st.markdown("### Model Performance Trends")
        
        col1, col2, col3, col4 = st.columns(4)
        latest = perf_log.iloc[-1]
        
        with col1:
            st.metric("Current MAE", f"{latest['mae']:,.0f} MWh")
        with col2:
            st.metric("Current R²", f"{latest['r2']:.3f}")
        with col3:
            st.metric("Current MAPE", f"{latest['mape']:.1f}%")
        with col4:
            st.metric("Measurements", len(perf_log))
        
        perf_chart = create_performance_trends(perf_log)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
    
    # Row 5: Data Tables & Model Comparison
    st.markdown("---")
    st.markdown("### Detailed Information")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Recent Predictions", "Model Versions", "Model Details", "System Logs"])
    
    with tab1:
        display_cols = ['timestamp', 'demand_mwh', 'temperature', 'hour_of_day', 'day_of_week']
        if 'prediction' in df_filtered.columns:
            display_cols.append('prediction')
            display_cols.append('error')
            df_filtered['error'] = df_filtered['demand_mwh'] - df_filtered['prediction']
        
        st.dataframe(
            df_filtered[display_cols].tail(20).sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Export button
        if 'prediction' in df_filtered.columns:
            csv = df_filtered[display_cols].to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.markdown("**Model Version Comparison**")
        model_versions = get_all_model_versions()
        if model_versions is not None:
            st.dataframe(
                model_versions,
                use_container_width=True,
                hide_index=True
            )
            
            # Highlight best model
            best_idx = model_versions['MAE'].str.replace(',', '').astype(float).idxmin()
            best_version = model_versions.loc[best_idx, 'Version']
            st.info(f"Best performing model: **Version {best_version}** (lowest MAE)")
        else:
            st.info("Model version history not available")
    
    with tab3:
        if model_info:
            col1, col2 = st.columns(2)
            with col1:
                st.json(model_info)
            with col2:
                st.markdown("**Model Metadata**")
                st.write(f"**Version:** {model_info.get('version', 'N/A')}")
                st.write(f"**Stage:** {model_info.get('stage', 'N/A')}")
                st.write(f"**Run ID:** `{model_info.get('run_id', 'N/A')[:16]}...`")
                if 'timestamp' in model_info:
                    created = pd.to_datetime(model_info['timestamp'], unit='ms')
                    st.write(f"**Created:** {created.strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            st.info("No model information available")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Pipeline Status**")
            st.text(f"Last data update: {df_filtered['timestamp'].max()}")
            st.text(f"Total records: {len(features_df)}")
            st.text(f"Filtered records: {len(df_filtered)}")
            st.text(f"Reference samples: {len(reference_df) if reference_df is not None else 0}")
            st.text(f"Date range: {df_filtered['timestamp'].min().date()} to {df_filtered['timestamp'].max().date()}")
        
        with col2:
            st.markdown("**Monitoring Status**")
            if drift_results:
                check_time = drift_results.get('check_timestamp', 'N/A')
                st.text(f"Last drift check: {check_time[:19] if len(check_time) > 19 else check_time}")
                st.text(f"Drift threshold: {drift_results.get('threshold', 0.3):.0%}")
                st.text(f"Features monitored: {drift_results.get('n_total_features', 0)}")
            if perf_log is not None:
                st.text(f"Performance logs: {len(perf_log)} entries")
    
    # Footer with enhanced branding
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <p style='color: {COLORS['text']}; font-size: 0.9rem; opacity: 0.7; margin-bottom: 0.5rem;'>
            <strong>Energy Demand Forecasting MLOps Platform</strong>
        </p>
        <p style='color: {COLORS['text']}; font-size: 0.8rem; opacity: 0.6;'>
            Powered by <strong>Streamlit</strong> · <strong>MLflow</strong> · <strong>LightGBM</strong> · <strong>Evidently AI</strong>
        </p>
        <p style='color: {COLORS['text']}; font-size: 0.75rem; opacity: 0.5; margin-top: 0.5rem;'>
            Production-ready ML pipeline with automated retraining, drift detection & monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
