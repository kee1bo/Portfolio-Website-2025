# /home/ashuran/Desktop/portfolio_website/app.py
from flask import Flask, render_template, redirect, url_for, abort
import math
import os # <--- ADD THIS IMPORT AT THE TOP

app = Flask(__name__)

# --- Data for the portfolio ---
# (Your skills, projects, education data remains here)
skills = [
    {"name": "Machine Learning", "icon": "brain", "level": 90, "category": "Expert"},
    {"name": "Natural Language Processing", "icon": "message-circle", "level": 85, "category": "Expert"},
    {"name": "Deep Learning", "icon": "cpu", "level": 88, "category": "Expert"},
    {"name": "Data Analysis", "icon": "database", "level": 92, "category": "Expert"},
    {"name": "Python", "icon": "code", "level": 95, "category": "Expert"},
    {"name": "TensorFlow/PyTorch", "icon": "layers", "level": 87, "category": "Expert"}
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
        "description": "Developed an advanced NLP model for multi-lingual sentiment analysis with 94% accuracy across 7 languages.",
        "tags": ["NLP", "PyTorch", "Transformers"],
        "thumbnail": "sentiment-analysis.jpg",
        "github": "https://github.com/kee1bo/sentiment-analysis",
        "medium": None,
        "youtube": None,
        "demo": None
    },
    {
        "id": "computer-vision",
        "title": "Computer Vision System",
        "description": "Built an object detection and tracking system for retail analytics, reducing inventory errors by 35%.",
        "tags": ["Computer Vision", "TensorFlow", "OpenCV"],
        "thumbnail": "computer-vision.jpg",
        "github": "https://github.com/kee1bo/computer-vision",
        "medium": None,
        "youtube": None,
        "demo": None
    },
    {
        "id": "predictive-analytics",
        "title": "Predictive Analytics Platform",
        "description": "Created a time-series forecasting solution for supply chain optimization, improving efficiency by 28%.",
        "tags": ["Time Series", "LSTM", "Prophet"],
        "thumbnail": "predictive-analytics.jpg",
        "github": "https://github.com/kee1bo/predictive-analytics",
        "medium": None,
        "youtube": None,
        "demo": None
    },
    {
        "id": "permanent-vs-determinant",
        "title": "Permanent vs Determinant",
        "description": "An exploratory study on matrix computations with applications in quantum computing and graph theory.",
        "tags": ["Linear Algebra", "Quantum Computing", "Graph Theory"],
        "thumbnail": "permanent-determinant.jpg",
        "github": "https://github.com/kee1bo/permanent-vs-determinant",
        "medium": None,
        "youtube": None,
        "demo": None
    },
    {
        "id": "rna-folding",
        "title": "mRNA Vaccine Degradation Prediction",
        "description": "Developed deep learning models to predict mRNA degradation rates for the Stanford OpenVaccine Kaggle competition.",
        "tags": ["Bioinformatics", "Deep Learning", "Kaggle", "RNA", "Regression"],
        "thumbnail": "rna-folding.jpg",
        "github": "https://github.com/kee1bo/kaggle-openvaccine",
        "medium": None,
        "youtube": None,
        "demo": "https://www.kaggle.com/c/stanford-covid-vaccine",
    }
]

education = [
     {
        "degree": "MTech in Artificial Intelligence and Data Science",
        "institution": "JNTUHCE Sulthanpur",
        "period": "2024 - 2026",
        "description": "Specializing in advanced machine learning algorithms, deep learning architectures, and large-scale data processing techniques with a focus on practical applications in industry and research.",
        "achievements": [
            "Ongoing research on transformer architectures for multimodal learning",
            "Working on a novel approach to few-shot learning in computer vision",
            "Published paper on efficient attention mechanisms in ICLR 2024",
            "Teaching assistant for Advanced Deep Learning course",
            "Awarded departmental scholarship for outstanding academic performance"
        ],
        "courses": [
            "Advanced Deep Learning",
            "Natural Language Processing",
            "Computer Vision Systems",
            "Reinforcement Learning",
            "Big Data Analytics"
        ],
        "icon": "graduation-cap"
    },
    {
        "degree": "BTech in Computer Science and Engineering",
        "institution": "Indian Institute of Technology, Indore",
        "period": "2016 - 2022",
        "description": "Gained strong foundations in computer science principles, algorithms, data structures, and software engineering practices with a focus on AI applications and system design.",
        "achievements": [
            "Graduated with honors, top 5% of class",
            "Led research project on neural network optimization",
            "Internship at Microsoft Research on reinforcement learning",
            "Received Best Undergraduate Thesis Award",
            "Published 2 papers in top-tier conferences"
        ],
        "courses": [
            "Data Structures and Algorithms",
            "Machine Learning Fundamentals",
            "Operating Systems",
            "Database Management Systems",
            "Software Engineering"
        ],
        "icon": "award"
    },
    {
        "degree": "AI Research Certificate Program",
        "institution": "Stanford University (Online)",
        "period": "2023",
        "description": "Intensive six-month program focused on cutting-edge AI research methodologies, advanced neural network architectures, and practical implementation of research papers.",
        "achievements": [
            "Completed with distinction (98% score)",
            "Implemented 5 state-of-the-art models from recent research papers",
            "Final project selected for showcase in program highlights",
            "Mentored by leading researchers in the field"
        ],
        "courses": [
            "Research Methods in AI",
            "Neural Network Architecture Design",
            "Paper Implementation Practicum",
            "AI Ethics and Responsibility"
        ],
        "icon": "book-open"
    }
]


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main index page."""
    return render_template('index.html',
                          skills=skills,
                          additional_skills=additional_skills,
                          projects=projects,
                          education=education)

@app.route('/projects/<project_id>/')
def project_detail(project_id):
    # ... (your existing project_detail logic) ...
    current_project = None
    current_index = -1

    for i, p in enumerate(projects):
        if p["id"] == project_id:
            current_project = p
            current_index = i
            break

    if current_project is None:
        print(f"Project with ID '{project_id}' not found.")
        abort(404)

    num_projects = len(projects)
    prev_index = (current_index - 1 + num_projects) % num_projects
    next_index = (current_index + 1) % num_projects

    prev_project = projects[prev_index]
    next_project = projects[next_index]

    template_name = f'projects/{project_id}.html'

    try:
        return render_template(template_name,
                               project=current_project,
                               prev_project=prev_project,
                               next_project=next_project)
    except Exception as e:
        print(f"Error rendering template {template_name}: {e}")
        abort(404)


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