#!/bin/bash

# Deployment script for Netlify
echo "Starting deployment..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Build the static site
echo "Building static site..."
python build.py

# Check if build was successful
if [ -d "build" ]; then
    echo "Build successful! Site generated in 'build' directory."
    ls -la build/
else
    echo "Build failed! No 'build' directory found."
    exit 1
fi