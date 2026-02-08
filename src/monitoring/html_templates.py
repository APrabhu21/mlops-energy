"""
HTML templates for drift and performance reports.
"""


def generate_drift_html(
    drift_share: float,
    details: dict,
    ref_stats: dict,
    cur_stats: dict,
    ref_size: int,
    cur_size: int,
    feature_columns: list
) -> str:
    """Generate custom HTML content for drift report."""
    
    # Determine status color and icon
    if drift_share > details['threshold']:
        status_color = "#dc3545"
        status_icon = "⚠️"
        status_text = "DRIFT DETECTED"
        recommendation = "Model retraining recommended"
    else:
        status_color = "#28a745"
        status_icon = "✓"
        status_text = "NO SIGNIFICANT DRIFT"
        recommendation = "Model is stable"
    
    # Build feature comparison table
    feature_rows = []
    for feature in feature_columns:
        ref_mean = ref_stats.get(feature, {}).get('mean', 0)
        cur_mean = cur_stats.get(feature, {}).get('mean', 0)
        ref_std = ref_stats.get(feature, {}).get('std', 0)
        cur_std = cur_stats.get(feature, {}).get('std', 0)
        
        # Calculate percent change
        pct_change = ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
        change_color = "#dc3545" if abs(pct_change) > 10 else "#6c757d"
        
        feature_rows.append(f"""
        <tr>
            <td>{feature}</td>
            <td>{ref_mean:.4f}</td>
            <td>{cur_mean:.4f}</td>
            <td style="color: {change_color}; font-weight: {'bold' if abs(pct_change) > 10 else 'normal'};">{pct_change:+.2f}%</td>
            <td>{ref_std:.4f}</td>
            <td>{cur_std:.4f}</td>
        </tr>
        """)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Drift Report - MLOps Energy Forecasting</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #f8f9fa;
                padding: 20px;
                color: #212529;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{
                color: #212529;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .subtitle {{
                color: #6c757d;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            .status-banner {{
                background: {status_color};
                color: white;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 30px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            .status-left {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            .status-icon {{
                font-size: 36px;
            }}
            .status-text {{
                font-size: 24px;
                font-weight: bold;
            }}
            .status-recommendation {{
                font-size: 14px;
                opacity: 0.9;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }}
            .metric-label {{
                color: #6c757d;
                font-size: 12px;
                text-transform: uppercase;
                margin-bottom: 8px;
            }}
            .metric-value {{
                font-size: 32px;
                font-weight: bold;
                color: #212529;
            }}
            .metric-unit {{
                font-size: 18px;
                color: #6c757d;
                margin-left: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            thead {{
                background: #f8f9fa;
            }}
            th {{
                padding: 12px;
                text-align: left;
                font-weight: 600;
                color: #495057;
                border-bottom: 2px solid #dee2e6;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #dee2e6;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .section-title {{
                font-size: 20px;
                font-weight: 600;
                margin: 30px 0 15px 0;
                color: #212529;
            }}
            .timestamp {{
                color: #6c757d;
                font-size: 12px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #dee2e6;
                text-align: center;
            }}
            .progress-bar {{
                width: 100%;
                height: 30px;
                background: #e9ecef;
                border-radius: 15px;
                overflow: hidden;
                margin-top: 10px;
                position: relative;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, {status_color}, {status_color}CC);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 14px;
                transition: width 0.3s ease;
            }}
            .threshold-marker {{
                position: absolute;
                top: 0;
                left: {details['threshold'] * 100}%;
                width: 2px;
                height: 100%;
                background: #212529;
                z-index: 10;
            }}
            .threshold-label {{
                position: absolute;
                top: -20px;
                left: {details['threshold'] * 100}%;
                transform: translateX(-50%);
                font-size: 10px;
                color: #6c757d;
                white-space: nowrap;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Data Drift Report</h1>
            <div class="subtitle">MLOps Energy Demand Forecasting System</div>
            
            <div class="status-banner">
                <div class="status-left">
                    <div class="status-icon">{status_icon}</div>
                    <div>
                        <div class="status-text">{status_text}</div>
                        <div class="status-recommendation">{recommendation}</div>
                    </div>
                </div>
                <div style="font-size: 48px; font-weight: bold;">{drift_share:.1%}</div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Drift Share</div>
                    <div class="metric-value">{drift_share:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Drifted Features</div>
                    <div class="metric-value">{details['n_drifted_features']}<span class="metric-unit">/ {details['n_total_features']}</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Threshold</div>
                    <div class="metric-value">{details['threshold']:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Reference Size</div>
                    <div class="metric-value">{ref_size:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Current Size</div>
                    <div class="metric-value">{cur_size:,}</div>
                </div>
            </div>
            
            <div class="section-title">Drift Progress</div>
            <div class="progress-bar">
                <div class="threshold-label">Threshold</div>
                <div class="threshold-marker"></div>
                <div class="progress-fill" style="width: {min(drift_share * 100, 100):.1f}%;">
                    {drift_share:.1%}
                </div>
            </div>
            
            <div class="section-title">Feature Statistics Comparison</div>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Reference Mean</th>
                        <th>Current Mean</th>
                        <th>Change</th>
                        <th>Reference Std</th>
                        <th>Current Std</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(feature_rows)}
                </tbody>
            </table>
            
            <div class="timestamp">
                Generated: {details['check_timestamp']}<br>
                Powered by Evidently AI & Custom MLOps Pipeline
            </div>
        </div>
    </body>
    </html>
    """
    
    return html
