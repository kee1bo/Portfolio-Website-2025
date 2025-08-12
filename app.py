# /home/ashuran/Desktop/portfolio_website/app.py
from flask import Flask, render_template, redirect, url_for, abort
import math
import os # <--- ADD THIS IMPORT AT THE TOP

app = Flask(__name__)

# --- Data for the portfolio ---
skills = [
    {"name": "Machine Learning", "icon": "brain", "category": "Advanced model development and deployment."},
    {"name": "Natural Language Processing", "icon": "message-circle", "category": "Building systems that understand text and speech."},
    {"name": "Deep Learning", "icon": "cpu", "category": "Expertise in neural networks and architectures."},
    {"name": "Data Analysis", "icon": "database", "category": "Extracting insights from complex datasets."},
    {"name": "Python", "icon": "code", "category": "Proficient in Python for AI and web development."},
    {"name": "TensorFlow/PyTorch", "icon": "layers", "category": "Experience with major deep learning frameworks."}
]

additional_skills = [
    'SQL', 'NoSQL', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'Git', 'CI/CD',
    'RESTful APIs', 'Flask', 'FastAPI', 'React', 'Data Visualization',
    'Scikit-learn', 'Pandas', 'NumPy', 'BERT', 'GPT', 'Computer Vision'
]

projects = [
    {
        "id": "sentiment-analysis",
        "title": "Sentiment Analysis Engine",
        "description": "Engineered a high-accuracy NLP model for multi-lingual sentiment analysis, achieving 94% accuracy across 7 languages.",
        "tags": ["NLP", "PyTorch", "Transformers"],
        "thumbnail": "sentiment-analysis.jpg",
        "github": "https://github.com/kee1bo/sentiment-analysis"
    },
    {
        "id": "computer-vision",
        "title": "Computer Vision System",
        "description": "Constructed an object detection system for retail analytics that reduced inventory errors by 35%.",
        "tags": ["Computer Vision", "TensorFlow", "OpenCV"],
        "thumbnail": "computer-vision.jpg",
        "github": "https://github.com/kee1bo/computer-vision"
    },
    {
        "id": "predictive-analytics",
        "title": "Predictive Analytics Platform",
        "description": "Developed a time-series forecasting solution to optimize supply chains, improving efficiency by 28%.",
        "tags": ["Time Series", "LSTM", "Prophet"],
        "thumbnail": "predictive-analytics.jpg",
        "github": "https://github.com/kee1bo/predictive-analytics"
    },
    {
        "id": "permanent-vs-determinant",
        "title": "Permanent vs Determinant",
        "description": "Conducted an exploratory study on matrix computations with applications in quantum computing and graph theory.",
        "tags": ["Linear Algebra", "Quantum Computing", "Graph Theory"],
        "thumbnail": "permanent-determinant.jpg",
        "github": "https://github.com/kee1bo/permanent-vs-determinant"
    },
    {
        "id": "rna-folding",
        "title": "mRNA Vaccine Degradation",
        "description": "Built deep learning models to predict mRNA degradation rates for the Stanford OpenVaccine Kaggle competition.",
        "tags": ["Bioinformatics", "Deep Learning", "Kaggle"],
        "thumbnail": "rna-folding.jpg",
        "github": "https://github.com/kee1bo/kaggle-openvaccine",
        "demo": "https://www.kaggle.com/c/stanford-covid-vaccine"
    }
]

education = [
     {
        "degree": "MTech in Artificial Intelligence and Data Science",
        "institution": "JNTUHCE Sulthanpur",
        "period": "2024 - 2026",
        "icon": "graduation-cap"
    },
    {
        "degree": "BTech in Computer Science and Engineering",
        "institution": "Indian Institute of Technology, Indore",
        "period": "2016 - 2022",
        "icon": "award"
    },
    {
        "degree": "Self Paced Online Learning",
        "institution": "Project Based Learning",
        "period": "2023-till now",
        "icon": "book-open"
    }
]


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main index page."""
    return render_template('index.html', projects=projects)

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html', skills=skills, education=education)

@app.route('/projects')
def projects_page():
    """Renders the projects page."""
    return render_template('projects.html', projects=projects)

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template('contact.html')

@app.route('/projects/<project_id>')
def project_detail(project_id):
    """Renders a single project page."""
    project = next((p for p in projects if p['id'] == project_id), None)
    if project is None:
        abort(404)
    return render_template(f'projects/{project_id}.html', project=project)


# --- Conditionally define the /toggle-theme route ---
if os.environ.get('FLASK_BUILD_MODE') != 'freeze':
    @app.route('/toggle-theme')
    def toggle_theme():
        """Placeholder for server-side theme logic.
           This route is excluded during static site generation.
        """
        return '', 204
# --- End conditional route ---


# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page."""
    return "Page Not Found", 404

@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 error page."""
    return "Internal Server Error", 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)