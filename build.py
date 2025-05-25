# /home/ashuran/Desktop/portfolio_website/build.py
import os
os.environ['FLASK_BUILD_MODE'] = 'freeze' # Set an environment variable

from flask_frozen import Freezer
from app import app  # This imports the 'app' instance from your app.py file

# Instantiate the Freezer
freezer = Freezer(app)

# You can keep this line, but the conditional route should be the primary fix
freezer.exclude_urls = ['/toggle-theme']
# freezer.log_url_sources = True # You can comment this out for now if it's not giving more info

print(f"DEBUG: FLASK_BUILD_MODE is set to: {os.environ.get('FLASK_BUILD_MODE')}")
print(f"DEBUG: Freezer instance: {freezer}")
print(f"DEBUG: URLs explicitly excluded: {freezer.exclude_urls}")


if __name__ == '__main__':
    print("Freezing site...")
    try:
        freezer.freeze()
        print("Site frozen successfully. Output in 'build' directory.")
    except Exception as e:
        print(f"An error occurred during freezing: {e}")
        import sys
        sys.exit(2) # Ensure a non-zero exit code on error