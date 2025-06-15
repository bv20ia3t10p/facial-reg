-- Create database if it doesn't exist
CREATE DATABASE facial_recognition;

-- Connect to the database
\c facial_recognition;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create clients table
CREATE TABLE clients (
    client_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_type VARCHAR(50) NOT NULL,  -- client1, client2, server
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table (employees)
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES clients(client_id),
    external_id VARCHAR(100) NOT NULL,  -- Directory name as employee ID
    department VARCHAR(100),
    position VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create emotion_records table
CREATE TABLE emotion_records (
    record_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id),
    client_id UUID REFERENCES clients(client_id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- FER+ emotion probabilities
    neutral FLOAT NOT NULL,
    happiness FLOAT NOT NULL,
    surprise FLOAT NOT NULL,
    sadness FLOAT NOT NULL,
    anger FLOAT NOT NULL,
    disgust FLOAT NOT NULL,
    fear FLOAT NOT NULL,
    contempt FLOAT NOT NULL,
    -- Additional metadata
    confidence FLOAT NOT NULL,
    location VARCHAR(100),
    device_id VARCHAR(100),
    session_id UUID
);

-- Create time_of_day_stats materialized view
CREATE MATERIALIZED VIEW time_of_day_stats AS
SELECT 
    user_id,
    client_id,
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    AVG(neutral) as avg_neutral,
    AVG(happiness) as avg_happiness,
    AVG(surprise) as avg_surprise,
    AVG(sadness) as avg_sadness,
    AVG(anger) as avg_anger,
    AVG(disgust) as avg_disgust,
    AVG(fear) as avg_fear,
    AVG(contempt) as avg_contempt,
    COUNT(*) as record_count
FROM emotion_records
GROUP BY user_id, client_id, EXTRACT(HOUR FROM timestamp);

-- Create daily_emotion_trends materialized view
CREATE MATERIALIZED VIEW daily_emotion_trends AS
SELECT 
    user_id,
    client_id,
    DATE(timestamp) as date,
    EXTRACT(DOW FROM timestamp) as day_of_week,
    AVG(neutral) as avg_neutral,
    AVG(happiness) as avg_happiness,
    AVG(surprise) as avg_surprise,
    AVG(sadness) as avg_sadness,
    AVG(anger) as avg_anger,
    AVG(disgust) as avg_disgust,
    AVG(fear) as avg_fear,
    AVG(contempt) as avg_contempt,
    COUNT(*) as record_count
FROM emotion_records
GROUP BY user_id, client_id, DATE(timestamp), EXTRACT(DOW FROM timestamp);

-- Create indexes
CREATE INDEX idx_users_client_id ON users(client_id);
CREATE INDEX idx_emotion_records_user_id ON emotion_records(user_id);
CREATE INDEX idx_emotion_records_client_id ON emotion_records(client_id);
CREATE INDEX idx_emotion_records_timestamp ON emotion_records(timestamp);
CREATE INDEX idx_emotion_records_location ON emotion_records(location);

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_emotion_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY time_of_day_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_emotion_trends;
END;
$$ LANGUAGE plpgsql;

-- Create function to get dominant emotion
CREATE OR REPLACE FUNCTION get_dominant_emotion(
    neutral FLOAT, happiness FLOAT, surprise FLOAT, 
    sadness FLOAT, anger FLOAT, disgust FLOAT,
    fear FLOAT, contempt FLOAT
) RETURNS VARCHAR AS $$
BEGIN
    RETURN (
        SELECT emotion
        FROM (VALUES 
            ('neutral', neutral),
            ('happiness', happiness),
            ('surprise', surprise),
            ('sadness', sadness),
            ('anger', anger),
            ('disgust', disgust),
            ('fear', fear),
            ('contempt', contempt)
        ) AS emotions(emotion, value)
        ORDER BY value DESC
        LIMIT 1
    ).emotion;
END;
$$ LANGUAGE plpgsql; 