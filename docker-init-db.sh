#!/bin/bash
set -e

# This script runs when PostgreSQL container first starts
# It creates both demo and live databases

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create demo database if it doesn't exist
    SELECT 'CREATE DATABASE crypto_arbitrage_demo'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'crypto_arbitrage_demo')\gexec

    -- Create live database if it doesn't exist
    SELECT 'CREATE DATABASE crypto_arbitrage_live'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'crypto_arbitrage_live')\gexec

    -- Grant all privileges to the postgres user
    GRANT ALL PRIVILEGES ON DATABASE crypto_arbitrage_demo TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE crypto_arbitrage_live TO $POSTGRES_USER;
EOSQL

echo "Demo and Live databases created successfully!"
