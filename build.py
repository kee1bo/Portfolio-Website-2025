from flask_frozen import Freezer
from app import app  # Assuming your Flask app instance is named 'app' in app.py

# Instantiate the Freezer
freezer = Freezer(app)

# Add this line to exclude the /toggle-theme URL
freezer.exclude_urls = ['/toggle-theme']

if __name__ == '__main__':
    print("Freezing site...")
    try:
        freezer.freeze()
        print("Site frozen successfully. Output in 'build' directory.")
    except Exception as e:
        print(f"An error occurred during freezing: {e}")
        import sys
        sys.exit(2) # Ensure a non-zero exit code on error