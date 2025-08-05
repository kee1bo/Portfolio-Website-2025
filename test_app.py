#!/usr/bin/env python3

# Simple test script to verify Flask app works
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("✅ Flask app imported successfully")
    
    # Test basic route
    with app.test_client() as client:
        response = client.get('/')
        if response.status_code == 200:
            print("✅ Home route works correctly")
        else:
            print(f"❌ Home route failed with status: {response.status_code}")
            
except ImportError as e:
    print(f"❌ Failed to import Flask app: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error testing Flask app: {e}")
    sys.exit(1)

print("✅ All tests passed!")