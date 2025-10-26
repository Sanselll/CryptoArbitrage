#!/bin/bash
# Script to remove large data files from Git history
# WARNING: This rewrites Git history! Only use if you haven't pushed to develop yet
# or coordinate with your team

echo "This script will remove large data files from Git history"
echo "WARNING: This rewrites history and requires force push!"
echo ""
echo "Files to be removed:"
echo "  - src/CryptoArbitrage.HistoricalCollector/data/"
echo "  - src/CryptoArbitrage.API/Data/backend_dumps/"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "git-filter-repo is not installed. Installing via pip..."
    pip3 install git-filter-repo
fi

# Create backup
echo "Creating backup branch..."
git branch backup-before-cleanup

# Remove the directories from history
echo "Removing data directories from Git history..."
git filter-repo --path src/CryptoArbitrage.HistoricalCollector/data --invert-paths --force
git filter-repo --path src/CryptoArbitrage.API/Data/backend_dumps --invert-paths --force

echo ""
echo "Cleanup complete!"
echo "Next steps:"
echo "1. Verify your repo looks correct"
echo "2. If you've already pushed to remote, you'll need to force push:"
echo "   git push origin feature/ML-Integration --force"
echo "3. If something went wrong, restore from backup:"
echo "   git reset --hard backup-before-cleanup"
