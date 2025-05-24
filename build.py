from flask_frozen import Freezer
from app import app

freezer = Freezer(app)
freezer.exclude_urls = ['/toggle-theme'] # Crucial line

if __name__ == '__main__':
    print("Freezing site...")
    try:
        freezer.freeze()
        print("Site frozen successfully. Output in 'build' directory.")
    except Exception as e:
        print(f"An error occurred during freezing: {e}")
        import sys
        sys.exit(2)