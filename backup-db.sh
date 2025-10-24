#!/bin/bash

# Database Backup Script for CryptoArbitrage
# Creates timestamped backups of the PostgreSQL database

# Configuration
DB_NAME="crypto_arbitrage"
DB_USER="sansel"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.sql"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo -e "${YELLOW}Starting database backup...${NC}"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "Backup file: $BACKUP_FILE"

# Perform backup using pg_dump
if /opt/homebrew/opt/postgresql@16/bin/pg_dump -U "$DB_USER" -F p -f "$BACKUP_FILE" "$DB_NAME"; then
    # Compress the backup
    gzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.gz"

    # Get file size
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)

    echo -e "${GREEN}✓ Backup successful!${NC}"
    echo "  File: $BACKUP_FILE"
    echo "  Size: $SIZE"

    # Keep only last 10 backups
    BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/*.sql.gz 2>/dev/null | wc -l)
    if [ "$BACKUP_COUNT" -gt 10 ]; then
        echo -e "${YELLOW}Cleaning up old backups (keeping last 10)...${NC}"
        ls -1t "$BACKUP_DIR"/*.sql.gz | tail -n +11 | xargs rm -f
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    fi

    exit 0
else
    echo -e "${RED}✗ Backup failed!${NC}"
    exit 1
fi
