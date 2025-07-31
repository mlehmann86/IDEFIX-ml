#!/bin/bash
# update_project.sh

set -e  # Exit immediately if any command exits with a non-zero status

# Sync private copy
echo "Pulling updates from private repository (origin)..."
git pull origin || { echo "Failed to pull from private repository. Resolve conflicts manually."; exit 1; }

# Fetch updates from public repository
echo "Pulling updates from public repository..."
git pull public master || { echo "Failed to pull from public repository. Resolve conflicts manually."; exit 1; }

# Update all submodules to their latest versions
#echo "Updating all submodules to their latest versions..."
#git submodule update --remote --merge || { echo "Failed to update submodules. Resolve conflicts manually."; exit 1; }

# Check for changes and commit them
#if ! git diff-index --quiet HEAD --; then
#    echo "Committing changes..."
#    git add .  # Stage changes
#    git commit -m "Sync private code and update public code including submodules to latest versions"
#    git push || { echo "Failed to push changes to private repository. Resolve conflicts manually."; exit 1; }
#else
#    echo "No changes to commit."
#fi

echo "Update completed successfully!"
