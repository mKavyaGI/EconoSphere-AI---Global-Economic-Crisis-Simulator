-- EconoSphere AI - Full Schema Creation Script
-- Applied directly via psql because asyncpg/Python 3.14 has a known
-- socket resolution bug on Windows that blocks Alembic's async engine.

-- ENUMS
DO $$ BEGIN
    CREATE TYPE eventcategory AS ENUM ('ECONOMIC','POLITICAL','ENVIRONMENTAL','FINANCIAL','HEALTH','ENERGY','TECHNOLOGY','SUPPLY_CHAIN');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE eventstatus AS ENUM ('DRAFT','ACTIVE','DISABLED','ARCHIVED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE scenariostatus AS ENUM ('DRAFT','PUBLISHED','ARCHIVED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- COUNTRIES
CREATE TABLE IF NOT EXISTS countries (
    iso3        VARCHAR(3) PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    region      VARCHAR(100),
    income_group VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS ix_countries_name   ON countries(name);
CREATE INDEX IF NOT EXISTS ix_countries_region ON countries(region);

-- COUNTRIES METADATA
CREATE TABLE IF NOT EXISTS countries_metadata (
    iso3        VARCHAR(3) PRIMARY KEY REFERENCES countries(iso3) ON DELETE CASCADE,
    capital     VARCHAR(255),
    population  INTEGER,
    currency    VARCHAR(50),
    flag_url    VARCHAR(1024),
    latitude    FLOAT,
    longitude   FLOAT
);

-- HISTORICAL INDICATORS
CREATE TABLE IF NOT EXISTS historical_indicators (
    id             SERIAL PRIMARY KEY,
    iso3           VARCHAR(3) REFERENCES countries(iso3) ON DELETE CASCADE,
    indicator_name VARCHAR(100),
    year           INTEGER,
    value          FLOAT NOT NULL,
    source         VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS ix_historical_indicators_iso3           ON historical_indicators(iso3);
CREATE INDEX IF NOT EXISTS ix_historical_indicators_indicator_name ON historical_indicators(indicator_name);
CREATE INDEX IF NOT EXISTS ix_historical_indicators_year           ON historical_indicators(year);

-- SCENARIOS
CREATE TABLE IF NOT EXISTS scenarios (
    id          SERIAL PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    description TEXT,
    author      VARCHAR(255),
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ,
    is_template BOOLEAN DEFAULT FALSE,
    tags        JSON,
    status      scenariostatus NOT NULL DEFAULT 'DRAFT'
);

-- SCENARIO VERSIONS
CREATE TABLE IF NOT EXISTS scenario_versions (
    id             SERIAL PRIMARY KEY,
    scenario_id    INTEGER NOT NULL REFERENCES scenarios(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    created_at     TIMESTAMPTZ DEFAULT now(),
    notes          TEXT
);

-- SCENARIO EVENTS
CREATE TABLE IF NOT EXISTS scenario_events (
    id                SERIAL PRIMARY KEY,
    version_id        INTEGER NOT NULL REFERENCES scenario_versions(id) ON DELETE CASCADE,
    parent_event_id   INTEGER REFERENCES scenario_events(id) ON DELETE SET NULL,
    title             VARCHAR(255) NOT NULL,
    category          eventcategory NOT NULL,
    status            eventstatus NOT NULL DEFAULT 'DRAFT',
    parameters        JSONB,
    "shockIntensity"  FLOAT DEFAULT 0.5,
    "propagationDelay" INTEGER DEFAULT 0,
    "recoveryRate"    FLOAT DEFAULT 0.1,
    "affectedNodes"   JSON,
    confidence        FLOAT DEFAULT 0.8,
    "simulationWeight" FLOAT DEFAULT 1.0
);

-- ALEMBIC VERSION TABLE (marks DB as fully migrated)
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL PRIMARY KEY
);
-- Insert a sentinel so Alembic knows the schema is up to date
INSERT INTO alembic_version (version_num)
VALUES ('scenario_status_manual')
ON CONFLICT DO NOTHING;

\echo 'Schema created successfully.'
