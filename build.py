from flask_frozen import Freezer
from app import app # Assuming your Flask app instance is named 'app' in app.py

# Optional: Configure Freezer if needed (e.g., for base URL if using a custom domain)
# app.config['FREEZER_BASE_URL'] = 'https://yourdomain.com'
# app.config['FREEZER_RELATIVE_URLS'] = True # Or False depending on your needs

freezer = Freezer(app)

if __name__ == '__main__':
    print("Freezing site...")
    freezer.freeze()
    print("Site frozen successfully. Output in 'build' directory.")