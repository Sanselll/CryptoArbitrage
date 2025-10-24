#!/bin/bash

# Database Restore Script for CryptoArbitrage
# Restores PostgreSQL database from a backup file

# Configuration
DB_NAME="crypto_arbitrage"
DB_USER="sansel"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backup file was provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: No backup file specified${NC}"
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh ./backups/*.sql.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Error: Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}WARNING: This will REPLACE the current database!${NC}"
echo "Database: $DB_NAME"
echo "Backup file: $BACKUP_FILE"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

echo -e "${YELLOW}Creating backup of current database before restore...${NC}"
./backup-db.sh

echo -e "${YELLOW}Dropping existing database...${NC}"
/opt/homebrew/opt/postgresql@16/bin/dropdb --if-exists -U "$DB_USER" "$DB_NAME"

echo -e "${YELLOW}Creating new database...${NC}"
/opt/homebrew/opt/postgresql@16/bin/createdb -U "$DB_USER" "$DB_NAME"

# Check if backup is compressed
if [[ "$BACKUP_FILE" == *.gz ]]; then
    echo -e "${YELLOW}Decompressing and restoring backup...${NC}"
    gunzip -c "$BACKUP_FILE" | /opt/homebrew/opt/postgresql@16/bin/psql -U "$DB_USER" "$DB_NAME"
else
    echo -e "${YELLOW}Restoring backup...${NC}"
    /opt/homebrew/opt/postgresql@16/bin/psql -U "$DB_USER" "$DB_NAME" < "$BACKUP_FILE"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database restored successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Restore failed!${NC}"
    exit 1
fi
