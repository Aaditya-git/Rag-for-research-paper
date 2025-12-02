#!/bin/bash

# Exit on first error
set -e

echo "Enter commit message:"
read msg

# Stage all changes
git add .

# Commit (allow empty to avoid errors)
git commit -m "$msg" || echo "Nothing to commit."

# Detect current branch
branch=$(git rev-parse --abbrev-ref HEAD)

echo "Pushing to branch: $branch"
git push origin "$branch"

echo "Push complete!"

