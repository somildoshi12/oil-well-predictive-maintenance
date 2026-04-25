-- Database initialization script for oil well predictive maintenance

CREATE TABLE IF NOT EXISTS wells (
    well_id VARCHAR(20) PRIMARY KEY,
    depth_ft INTEGER NOT NULL,
    install_date DATE,
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    record_id BIGINT PRIMARY KEY,
    well_id VARCHAR(20) REFERENCES wells(well_id),
    timestamp TIMESTAMP NOT NULL,
    depth_ft INTEGER,
    pump_pressure_psi FLOAT,
    flow_rate_bpd FLOAT,
    vibration_mm_s FLOAT,
    temperature_f FLOAT,
    torque_ft_lbs FLOAT,
    motor_current_amp FLOAT,
    oil_viscosity_cp FLOAT,
    gas_oil_ratio FLOAT,
    rpm FLOAT,
    hours_since_last_maintenance FLOAT,
    cumulative_operating_hours FLOAT,
    maintenance_required INTEGER,
    failure_type VARCHAR(30),
    days_to_failure FLOAT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sensor_well_time ON sensor_readings(well_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS maintenance_events (
    id SERIAL PRIMARY KEY,
    well_id VARCHAR(20) REFERENCES wells(well_id),
    scheduled_date DATE,
    completed_date DATE,
    maintenance_type VARCHAR(50),
    failure_type_addressed VARCHAR(30),
    technician VARCHAR(100),
    notes TEXT,
    status VARCHAR(20) DEFAULT 'scheduled',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    well_id VARCHAR(20) REFERENCES wells(well_id),
    record_id BIGINT,
    timestamp TIMESTAMP,
    predicted_failure INTEGER,
    confidence_score FLOAT,
    days_to_failure_pred FLOAT,
    anomaly_score FLOAT,
    is_anomaly BOOLEAN,
    predicted_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_well ON model_predictions(well_id, predicted_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    well_id VARCHAR(20) REFERENCES wells(well_id),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    days_to_failure FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    acknowledged_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
