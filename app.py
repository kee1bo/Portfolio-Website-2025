# /home/ashuran/Desktop/portfolio_website/app.py
from flask import Flask, render_template, redirect, url_for, abort, request, jsonify, flash
from datetime import datetime, timedelta
import math
import os
import json
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-portfolio-site')

# Database initialization
def init_db():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    # Blog posts table
    c.execute('''CREATE TABLE IF NOT EXISTS blog_posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  slug TEXT UNIQUE NOT NULL,
                  content TEXT NOT NULL,
                  excerpt TEXT,
                  featured_image TEXT,
                  tags TEXT,
                  published INTEGER DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  views INTEGER DEFAULT 0,
                  reading_time INTEGER DEFAULT 5)''')
    
    # GitHub repositories table
    c.execute('''CREATE TABLE IF NOT EXISTS github_repos
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  description TEXT,
                  html_url TEXT,
                  language TEXT,
                  stars INTEGER DEFAULT 0,
                  forks INTEGER DEFAULT 0,
                  updated_at TIMESTAMP,
                  featured INTEGER DEFAULT 0)''')
    
    # YouTube videos table
    c.execute('''CREATE TABLE IF NOT EXISTS youtube_videos
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  video_id TEXT UNIQUE NOT NULL,
                  description TEXT,
                  thumbnail_url TEXT,
                  published_at TIMESTAMP,
                  views INTEGER DEFAULT 0,
                  duration TEXT,
                  featured INTEGER DEFAULT 0)''')
    
    # Analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS analytics
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  page_path TEXT NOT NULL,
                  visitor_ip TEXT,
                  user_agent TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# --- Enhanced Data for the portfolio ---
skills = [
    {"name": "Machine Learning", "icon": "brain", "level": 90, "category": "Expert", "projects": ["Sentiment Analysis", "Predictive Analytics"], "years": 4},
    {"name": "Natural Language Processing", "icon": "message-circle", "level": 85, "category": "Expert", "projects": ["Sentiment Analysis", "Text Summarization"], "years": 3},
    {"name": "Deep Learning", "icon": "cpu", "level": 88, "category": "Expert", "projects": ["Computer Vision", "Neural Networks"], "years": 3},
    {"name": "Data Analysis", "icon": "database", "level": 92, "category": "Expert", "projects": ["All Projects"], "years": 5},
    {"name": "Python", "icon": "code", "level": 95, "category": "Expert", "projects": ["All Projects"], "years": 6},
    {"name": "TensorFlow/PyTorch", "icon": "layers", "level": 87, "category": "Expert", "projects": ["Computer Vision", "NLP Models"], "years": 3}
]

additional_skills = [
    'SQL', 'NoSQL', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'Git', 'CI/CD',
    'RESTful APIs', 'Flask', 'FastAPI', 'React', 'Data Visualization',
    'Scikit-learn', 'Pandas', 'NumPy', 'BERT', 'GPT', 'Computer Vision',
    'MLOps', 'Apache Spark', 'Kafka', 'Redis', 'PostgreSQL', 'MongoDB'
]

projects = [
    {
        "id": "sentiment-analysis",
        "title": "Sentiment Analysis Engine",
        "description": "Developed an advanced NLP model for multi-lingual sentiment analysis with 94% accuracy across 7 languages.",
        "long_description": "A sophisticated natural language processing system that analyzes emotional tone in text across multiple languages using state-of-the-art transformer architectures.",
        "tags": ["NLP", "PyTorch", "Transformers", "Multi-lingual", "Production"],
        "thumbnail": "sentiment-analysis.jpg",
        "images": ["sentiment-demo.jpg", "architecture.jpg", "results.jpg"],
        "github": "https://github.com/kee1bo/sentiment-analysis",
        "live_demo": "https://sentiment-demo.vercel.app",
        "medium": "https://medium.com/@kee1bo/building-multilingual-sentiment-analysis",
        "youtube": "dQw4w9WgXcQ",
        "technologies": ["Python", "PyTorch", "Transformers", "FastAPI", "Docker", "AWS"],
        "status": "Production",
        "completion_date": "2024-01-15",
        "featured": True,
        "metrics": {
            "accuracy": 94.2,
            "languages": 7,
            "requests_per_day": 10000,
            "model_size": "110MB"
        }
    },
    {
        "id": "computer-vision",
        "title": "Computer Vision System",
        "description": "Built an object detection and tracking system for retail analytics, reducing inventory errors by 35%.",
        "long_description": "An end-to-end computer vision solution for retail environments featuring real-time object detection, tracking, and analytics dashboard.",
        "tags": ["Computer Vision", "TensorFlow", "OpenCV", "Real-time", "Analytics"],
        "thumbnail": "computer-vision.jpg",
        "images": ["cv-demo.jpg", "dashboard.jpg", "detection.jpg"],
        "github": "https://github.com/kee1bo/computer-vision",
        "live_demo": "https://cv-retail-demo.herokuapp.com",
        "youtube": "dQw4w9WgXcQ",
        "technologies": ["Python", "TensorFlow", "OpenCV", "Flask", "PostgreSQL", "Docker"],
        "status": "Production",
        "completion_date": "2023-11-20",
        "featured": True,
        "metrics": {
            "accuracy": 92.8,
            "fps": 30,
            "objects_detected": 25,
            "improvement": "35% error reduction"
        }
    },
    {
        "id": "predictive-analytics",
        "title": "Predictive Analytics Platform",
        "description": "Created a time-series forecasting solution for supply chain optimization, improving efficiency by 28%.",
        "long_description": "A comprehensive analytics platform that leverages machine learning for supply chain forecasting and optimization.",
        "tags": ["Time Series", "LSTM", "Prophet", "Supply Chain", "Optimization"],
        "thumbnail": "predictive-analytics.jpg",
        "images": ["forecasting.jpg", "dashboard.jpg", "optimization.jpg"],
        "github": "https://github.com/kee1bo/predictive-analytics",
        "live_demo": "https://predict-supply.herokuapp.com",
        "technologies": ["Python", "LSTM", "Prophet", "Streamlit", "AWS", "Apache Spark"],
        "status": "Production",
        "completion_date": "2023-09-10",
        "featured": True,
        "metrics": {
            "accuracy": 89.5,
            "efficiency_gain": "28%",
            "cost_savings": "$2.4M annually",
            "predictions_daily": 50000
        }
    },
    {
        "id": "permanent-vs-determinant",
        "title": "Permanent vs Determinant",
        "description": "An exploratory study on matrix computations with applications in quantum computing and graph theory.",
        "long_description": "Research project exploring the computational complexity differences between matrix permanent and determinant calculations with applications in quantum algorithms.",
        "tags": ["Linear Algebra", "Quantum Computing", "Graph Theory", "Research", "Mathematics"],
        "thumbnail": "permanent-determinant.jpg",
        "images": ["theory.jpg", "algorithms.jpg", "results.jpg"],
        "github": "https://github.com/kee1bo/permanent-vs-determinant",
        "medium": "https://medium.com/@kee1bo/permanent-vs-determinant-study",
        "technologies": ["Python", "NumPy", "Qiskit", "Jupyter", "LaTeX", "SciPy"],
        "status": "Research",
        "completion_date": "2022-05-15",
        "featured": False,
        "metrics": {
            "algorithms_implemented": 8,
            "performance_gain": "15x faster",
            "complexity_reduction": "O(n!) to O(n³)",
            "citations": 12
        }
    },
    {
        "id": "rna-folding",
        "title": "mRNA Vaccine Degradation Prediction",
        "description": "Developed deep learning models to predict mRNA degradation rates for the Stanford OpenVaccine Kaggle competition.",
        "long_description": "Deep learning approach to predict mRNA degradation patterns, contributing to vaccine stability research during the COVID-19 pandemic.",
        "tags": ["Bioinformatics", "Deep Learning", "Kaggle", "RNA", "Regression", "Healthcare"],
        "thumbnail": "rna-folding.jpg",
        "images": ["rna-structure.jpg", "model-arch.jpg", "predictions.jpg"],
        "github": "https://github.com/kee1bo/kaggle-openvaccine",
        "demo": "https://www.kaggle.com/c/stanford-covid-vaccine",
        "medium": "https://medium.com/@kee1bo/predicting-mrna-degradation",
        "technologies": ["Python", "TensorFlow", "BioPython", "Pandas", "Seaborn", "Jupyter"],
        "status": "Competition",
        "completion_date": "2020-12-01",
        "featured": False,
        "metrics": {
            "kaggle_rank": "Top 15%",
            "mcrmse_score": 0.18,
            "models_ensemble": 5,
            "data_points": "3M+ sequences"
        }
    }
]

education = [
     {
        "degree": "MTech in Artificial Intelligence and Data Science",
        "institution": "JNTUHCE Sulthanpur",
        "period": "2024 - 2026",
        "status": "Current",
        "description": "Specializing in advanced machine learning algorithms, deep learning architectures, and large-scale data processing techniques with a focus on practical applications in industry and research.",
        "achievements": [
            "Ongoing research on transformer architectures for multimodal learning",
            "Working on a novel approach for ARC Competition in Kaggle",
            "Teaching Assistant for Machine Learning Fundamentals course"
        ],
        "courses": [
            "Advanced Deep Learning",
            "Natural Language Processing",
            "Computer Vision Systems",
            "Reinforcement Learning",
            "Big Data Analytics",
            "AI Ethics and Explainability"
        ],
        "icon": "graduation-cap",
        "gpa": "9.2/10",
        "thesis": "Multimodal Learning for Cross-Domain Transfer"
    },
    {
        "degree": "BTech in Computer Science and Engineering",
        "institution": "Indian Institute of Technology, Indore",
        "period": "2016 - 2022",
        "status": "Completed",
        "description": "Gained strong foundations in computer science principles, algorithms, data structures, and software engineering practices with a focus on AI applications and system design.",
        "achievements": [
            "Led research project on 'Permanent vs Determinant' exploring matrix computations",
            "Dean's List for 4 consecutive semesters",
            "Founded AI/ML study group with 150+ members"
        ],
        "courses": [
            "Data Structures and Algorithms",
            "Machine Learning Fundamentals",
            "Operating Systems",
            "Database Management Systems",
            "Software Engineering",
            "Distributed Systems"
        ],
        "icon": "award",
        "gpa": "8.7/10",
        "final_project": "Distributed Machine Learning Framework"
    },
    {
        "degree": "Self Paced Online Learning",
        "institution": "Project Based Learning",
        "period": "2023-till now",
        "status": "Ongoing",
        "description": "Focused on cutting-edge AI research methodologies, advanced neural network architectures, and practical implementation of research papers.",
        "achievements": [
            "Implemented 15+ state-of-the-art models from recent research papers",
            "Contributed to 5+ open-source ML libraries",
            "Completed 25+ specialized AI/ML courses"
        ],
        "courses": [
            "Research Methods in AI",
            "Neural Network Architecture Design",
            "Paper Implementation Practicum",
            "AI Ethics and Responsibility",
            "MLOps and Production Systems"
        ],
        "icon": "book-open",
        "certifications": ["AWS ML Specialty", "Google Cloud ML Engineer", "Azure AI Engineer"]
    }
]

# Experience data
experience = [
    {
        "title": "Senior AI Engineer",
        "company": "TechCorp Solutions",
        "period": "2023 - Present",
        "location": "Remote",
        "description": "Leading AI initiatives and developing scalable machine learning solutions for enterprise clients.",
        "achievements": [
            "Developed ML pipelines serving 10M+ requests daily",
            "Reduced model inference time by 60% through optimization",
            "Led team of 5 engineers on computer vision projects"
        ],
        "technologies": ["Python", "TensorFlow", "AWS", "Kubernetes", "MLflow"]
    },
    {
        "title": "ML Research Intern",
        "company": "AI Research Lab",
        "period": "2022 - 2023",
        "location": "Bangalore, India",
        "description": "Conducted research on transformer architectures and their applications in NLP tasks.",
        "achievements": [
            "Published 2 papers in top-tier AI conferences",
            "Developed novel attention mechanism improving BERT performance by 8%",
            "Collaborated with international research teams"
        ],
        "technologies": ["PyTorch", "Transformers", "CUDA", "Weights & Biases"]
    }
]

# --- Analytics Functions ---
def track_page_visit(page_path, request):
    try:
        conn = sqlite3.connect('portfolio.db')
        c = conn.cursor()
        visitor_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        c.execute("INSERT INTO analytics (page_path, visitor_ip, user_agent) VALUES (?, ?, ?)",
                 (page_path, visitor_ip, user_agent))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Analytics error: {e}")

def get_blog_posts(published_only=True, limit=None):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    query = "SELECT * FROM blog_posts"
    if published_only:
        query += " WHERE published = 1"
    query += " ORDER BY created_at DESC"
    if limit:
        query += f" LIMIT {limit}"
    
    c.execute(query)
    posts = []
    for row in c.fetchall():
        posts.append({
            'id': row[0], 'title': row[1], 'slug': row[2], 'content': row[3],
            'excerpt': row[4], 'featured_image': row[5], 'tags': row[6].split(',') if row[6] else [],
            'published': row[7], 'created_at': row[8], 'updated_at': row[9],
            'views': row[10], 'reading_time': row[11]
        })
    
    conn.close()
    return posts

def get_featured_content():
    """Get featured projects, blogs, and videos for homepage"""
    featured_projects = [p for p in projects if p.get('featured', False)]
    recent_blogs = get_blog_posts(limit=3)
    
    # Mock YouTube data (replace with actual API integration)
    featured_videos = [
        {
            'title': 'Building Scalable ML Pipelines',
            'video_id': 'dQw4w9WgXcQ',
            'thumbnail': 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
            'views': 15420,
            'duration': '12:34'
        },
        {
            'title': 'Deep Learning for NLP',
            'video_id': 'dQw4w9WgXcQ',
            'thumbnail': 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
            'views': 8930,
            'duration': '18:45'
        }
    ]
    
    return {
        'projects': featured_projects[:3],
        'blogs': recent_blogs,
        'videos': featured_videos
    }

# --- Flask Routes ---

@app.route('/')
def index():
    """Enhanced homepage with featured content"""
    track_page_visit('/', request)
    featured_content = get_featured_content()
    
    return render_template('index.html',
                          skills=skills,
                          additional_skills=additional_skills,
                          projects=projects,
                          education=education,
                          experience=experience,
                          featured_content=featured_content)

@app.route('/projects/')
def projects_overview():
    """Projects overview page with filtering and search"""
    track_page_visit('/projects/', request)
    
    # Get filter parameters
    tech_filter = request.args.get('tech', '')
    status_filter = request.args.get('status', '')
    search_query = request.args.get('search', '')
    
    filtered_projects = projects
    
    if tech_filter:
        filtered_projects = [p for p in filtered_projects 
                           if tech_filter.lower() in [t.lower() for t in p.get('technologies', [])]]
    
    if status_filter:
        filtered_projects = [p for p in filtered_projects 
                           if p.get('status', '').lower() == status_filter.lower()]
    
    if search_query:
        filtered_projects = [p for p in filtered_projects 
                           if search_query.lower() in p['title'].lower() or 
                              search_query.lower() in p['description'].lower()]
    
    # Get unique technologies and statuses for filters
    all_technologies = set()
    all_statuses = set()
    for project in projects:
        all_technologies.update(project.get('technologies', []))
        all_statuses.add(project.get('status', 'Unknown'))
    
    return render_template('projects/overview.html',
                          projects=filtered_projects,
                          all_technologies=sorted(all_technologies),
                          all_statuses=sorted(all_statuses),
                          current_filters={
                              'tech': tech_filter,
                              'status': status_filter,
                              'search': search_query
                          })

@app.route('/projects/<project_id>/')
def project_detail(project_id):
    """Enhanced project detail page"""
    track_page_visit(f'/projects/{project_id}/', request)
    
    current_project = None
    current_index = -1

    for i, p in enumerate(projects):
        if p["id"] == project_id:
            current_project = p
            current_index = i
            break

    if current_project is None:
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

@app.route('/blog/')
def blog_index():
    """Blog listing page"""
    track_page_visit('/blog/', request)
    
    page = request.args.get('page', 1, type=int)
    tag_filter = request.args.get('tag', '')
    search_query = request.args.get('search', '')
    
    posts = get_blog_posts()
    
    if tag_filter:
        posts = [p for p in posts if tag_filter.lower() in [t.lower() for t in p['tags']]]
    
    if search_query:
        posts = [p for p in posts 
                if search_query.lower() in p['title'].lower() or 
                   search_query.lower() in p['content'].lower()]
    
    # Pagination
    per_page = 6
    start = (page - 1) * per_page
    end = start + per_page
    paginated_posts = posts[start:end]
    
    # Get all tags for filter
    all_tags = set()
    for post in get_blog_posts():
        all_tags.update(post['tags'])
    
    return render_template('blog/index.html',
                          posts=paginated_posts,
                          all_tags=sorted(all_tags),
                          current_page=page,
                          total_pages=math.ceil(len(posts) / per_page),
                          current_filters={
                              'tag': tag_filter,
                              'search': search_query
                          })

@app.route('/blog/<slug>/')
def blog_post(slug):
    """Individual blog post page"""
    track_page_visit(f'/blog/{slug}/', request)
    
    # Increment view count
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("UPDATE blog_posts SET views = views + 1 WHERE slug = ?", (slug,))
    conn.commit()
    
    c.execute("SELECT * FROM blog_posts WHERE slug = ? AND published = 1", (slug,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        abort(404)
    
    post = {
        'id': row[0], 'title': row[1], 'slug': row[2], 'content': row[3],
        'excerpt': row[4], 'featured_image': row[5], 'tags': row[6].split(',') if row[6] else [],
        'published': row[7], 'created_at': row[8], 'updated_at': row[9],
        'views': row[10], 'reading_time': row[11]
    }
    
    # Get related posts
    related_posts = get_blog_posts(limit=3)
    related_posts = [p for p in related_posts if p['slug'] != slug][:3]
    
    return render_template('blog/post.html', 
                          post=post, 
                          related_posts=related_posts)

@app.route('/about/')
def about():
    """Detailed about page"""
    track_page_visit('/about/', request)
    return render_template('about.html',
                          skills=skills,
                          education=education,
                          experience=experience)

@app.route('/contact/', methods=['GET', 'POST'])
def contact():
    """Enhanced contact page with form handling"""
    track_page_visit('/contact/', request)
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject', 'Portfolio Contact')
        message = request.form.get('message')
        
        # Here you would typically send an email or save to database
        # For now, just flash a success message
        flash('Thank you for your message! I\'ll get back to you soon.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/resume/')
def resume():
    """Resume/CV page"""
    track_page_visit('/resume/', request)
    return render_template('resume.html',
                          skills=skills,
                          education=education,
                          experience=experience,
                          projects=projects[:3])  # Featured projects only

@app.route('/api/analytics')
def analytics_api():
    """Simple analytics API"""
    try:
        conn = sqlite3.connect('portfolio.db')
        c = conn.cursor()
        
        # Get recent page views
        c.execute("""
            SELECT page_path, COUNT(*) as views 
            FROM analytics 
            WHERE timestamp > datetime('now', '-30 days')
            GROUP BY page_path 
            ORDER BY views DESC
        """)
        
        page_views = [{'page': row[0], 'views': row[1]} for row in c.fetchall()]
        
        # Get total views
        c.execute("SELECT COUNT(*) FROM analytics")
        total_views = c.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'page_views': page_views,
            'total_views': total_views
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Conditionally define the /toggle-theme route ---
if os.environ.get('FLASK_BUILD_MODE') != 'freeze':
    @app.route('/toggle-theme')
    def toggle_theme():
        """Placeholder for server-side theme logic."""
        return '', 204

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page."""
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 error page."""
    return render_template('errors/500.html'), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)