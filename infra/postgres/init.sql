-- Database initialization script for Energy Demand MLOps

-- Raw demand data from EIA
CREATE TABLE IF NOT EXISTS raw_demand (
    id SERIAL PRIMARY KEY,
    period TIMESTAMP NOT NULL,
    respondent VARCHAR(10) NOT NULL,
    value FLOAT NOT NULL,                -- MWh
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(period, respondent)
);
CREATE INDEX idx_demand_period ON raw_demand(period);

-- Raw weather data from Open-Meteo
CREATE TABLE IF NOT EXISTS raw_weather (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    temperature_2m FLOAT,
    relative_humidity_2m FLOAT,
    dewpoint_2m FLOAT,
    apparent_temperature FLOAT,
    precipitation FLOAT,
    snowfall FLOAT,
    cloud_cover FLOAT,
    wind_speed_10m FLOAT,
    wind_gusts_10m FLOAT,
    surface_pressure FLOAT,
    shortwave_radiation FLOAT,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(timestamp)
);
CREATE INDEX idx_weather_timestamp ON raw_weather(timestamp);

-- Feature-engineered dataset (joined demand + weather + time features)
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL UNIQUE,
    -- target
    demand_mwh FLOAT NOT NULL,
    -- weather features
    temperature FLOAT,
    apparent_temperature FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    cloud_cover FLOAT,
    solar_radiation FLOAT,
    precipitation FLOAT,
    -- time features
    hour_of_day INT,
    day_of_week INT,
    month INT,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    -- lag features
    demand_lag_1h FLOAT,
    demand_lag_24h FLOAT,
    demand_lag_168h FLOAT,           -- 1 week ago
    demand_rolling_mean_24h FLOAT,
    demand_rolling_std_24h FLOAT,
    -- temperature interaction
    temp_squared FLOAT,              -- captures nonlinear heating/cooling
    cooling_degree_hours FLOAT,      -- max(0, temp - 18.3)
    heating_degree_hours FLOAT,      -- max(0, 18.3 - temp)
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_features_timestamp ON features(timestamp);

-- Prediction log (for monitoring)
CREATE TABLE IF NOT EXISTS prediction_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,         -- prediction target time
    predicted_demand FLOAT NOT NULL,
    actual_demand FLOAT,                  -- filled in later when actual arrives
    model_version VARCHAR(50),
    prediction_made_at TIMESTAMP DEFAULT NOW(),
    features_json JSONB                   -- snapshot of input features
);
CREATE INDEX idx_predictions_timestamp ON prediction_log(timestamp);

-- Drift detection results
CREATE TABLE IF NOT EXISTS drift_log (
    id SERIAL PRIMARY KEY,
    check_timestamp TIMESTAMP DEFAULT NOW(),
    dataset_drift_detected BOOLEAN,
    drift_share FLOAT,                    -- % of features drifted
    n_drifted_features INT,
    n_total_features INT,
    drift_details JSONB,                  -- per-feature drift scores
    triggered_retrain BOOLEAN DEFAULT FALSE
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50),
    mae FLOAT,
    rmse FLOAT,
    mape FLOAT,
    r2 FLOAT,
    evaluation_period_start TIMESTAMP,
    evaluation_period_end TIMESTAMP
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlops;
