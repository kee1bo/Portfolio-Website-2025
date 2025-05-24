# # from flask import Flask, render_template, redirect, url_for
# # import math

# # app = Flask(__name__)

# # # Make math functions available to templates
# # @app.template_filter('cos')
# # def cos_filter(value):
# #     return math.cos(value)

# # @app.template_filter('sin')
# # def sin_filter(value):
# #     return math.sin(value)

# # # Data for the portfolio
# # skills = [
# #     {"name": "Machine Learning", "icon": "brain", "level": 90, "category": "Expert"},
# #     {"name": "Natural Language Processing", "icon": "code", "level": 85, "category": "Expert"},
# #     {"name": "Deep Learning", "icon": "cpu", "level": 88, "category": "Expert"},
# #     {"name": "Data Analysis", "icon": "database", "level": 92, "category": "Expert"},
# #     {"name": "Python", "icon": "code", "level": 95, "category": "Expert"},
# #     {"name": "TensorFlow/PyTorch", "icon": "brain", "level": 87, "category": "Expert"}
# # ]

# # additional_skills = [
# #     'SQL', 'NoSQL', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'Git', 'CI/CD', 
# #     'RESTful APIs', 'Flask', 'FastAPI', 'React', 'Data Visualization', 
# #     'Scikit-learn', 'Pandas', 'NumPy', 'BERT', 'GPT', 'Computer Vision'
# # ]

# # projects = [
# #     {
# #         "id": "sentiment-analysis",
# #         "title": "Sentiment Analysis Engine",
# #         "description": "Developed an advanced NLP model for multi-lingual sentiment analysis with 94% accuracy across 7 languages.",
# #         "tags": ["NLP", "PyTorch", "Transformers"],
# #         "thumbnail": "sentiment-analysis.jpg",
# #         "github": "https://github.com/neerudisaivikas/sentiment-analysis",
# #         "medium": "https://medium.com/@neerudisaivikas/sentiment-analysis",
# #         "youtube": "https://youtube.com/watch?v=example",
# #         "demo": "https://demo.sentiment-analysis.com"
# #     },
# #     {
# #         "id": "computer-vision",
# #         "title": "Computer Vision System",
# #         "description": "Built an object detection and tracking system for retail analytics, reducing inventory errors by 35%.",
# #         "tags": ["Computer Vision", "TensorFlow", "OpenCV"],
# #         "thumbnail": "computer-vision.jpg",
# #         "github": "https://github.com/neerudisaivikas/computer-vision",
# #         "medium": "https://medium.com/@neerudisaivikas/computer-vision",
# #         "youtube": "https://youtube.com/watch?v=example2",
# #         "demo": None
# #     },
# #     {
# #         "id": "predictive-analytics",
# #         "title": "Predictive Analytics Platform",
# #         "description": "Created a time-series forecasting solution for supply chain optimization, improving efficiency by 28%.",
# #         "tags": ["Time Series", "LSTM", "Prophet"],
# #         "thumbnail": "predictive-analytics.jpg",
# #         "github": "https://github.com/neerudisaivikas/predictive-analytics",
# #         "medium": "https://medium.com/@neerudisaivikas/predictive-analytics",
# #         "youtube": None,
# #         "demo": "https://demo.predictive-analytics.com"
# #     },
# #         {
# #         "id": "permanent-vs-determinant",
# #         "title": "Permanent vs Determinant",
# #         "description": "An exploratory study on matrix computations with applications in quantum computing and graph theory.",
# #         "tags": ["Linear Algebra", "Quantum Computing", "Graph Theory"],
# #         "thumbnail": "permanent-determinant.jpg",
# #         "github": "https://github.com/neerudisaivikas/permanent-vs-determinant",
# #         "medium": None,
# #         "youtube": "https://youtube.com/watch?v=matrix-computation",
# #         "demo": "https://permanent-vs-determinant-demo.com"
# #     }
# # ]

# # education = [
# #     {
# #         "degree": "MTech in Artificial Intelligence and Data Science",
# #         "institution": "JNTUHCE Sulthanpur",
# #         "period": "2024 - 2026",
# #         "description": "Specializing in advanced machine learning algorithms, deep learning architectures, and large-scale data processing techniques.",
# #         "achievements": [
# #             "Ongoing research on transformer architectures for multimodal learning",
# #             "Working on a novel approach to few-shot learning in computer vision",
# #             "Published paper on efficient attention mechanisms in ICLR 2024"
# #         ],
# #         "icon": "calendar"
# #     },
# #     {
# #         "degree": "BTech in Computer Science and Engineering",
# #         "institution": "Indian Institute of Technology, Indore",
# #         "period": "2016 - 2022",
# #         "description": "Gained strong foundations in computer science principles, algorithms, data structures, and software engineering practices with a focus on AI applications.",
# #         "achievements": [
# #             "Graduated with honors, top 5% of class",
# #             "Led research project on neural network optimization",
# #             "Internship at Microsoft Research on reinforcement learning"
# #         ],
# #         "icon": "award"
# #     }
# # ]

# # character_profile = {
# #     "name": "AI Engineer",
# #     "level": 32,
# #     "class": "Data Scientist",
# #     "stats": {
# #         "intelligence": 95,
# #         "creativity": 88,
# #         "speed": 82,
# #         "persistence": 90,
# #         "collaboration": 85
# #     },
# #     "special_abilities": [
# #         "Pattern Recognition", 
# #         "Algorithm Mastery", 
# #         "Data Intuition",
# #         "Code Optimization"
# #     ]
# # }

# # education = [
# #     {
# #         "degree": "MTech in Artificial Intelligence and Data Science",
# #         "institution": "JNTUHCE Sulthanpur",
# #         "period": "2024 - 2026",
# #         "description": "Specializing in advanced machine learning algorithms, deep learning architectures, and large-scale data processing techniques with a focus on practical applications in industry and research.",
# #         "achievements": [
# #             "Ongoing research on transformer architectures for multimodal learning",
# #             "Working on a novel approach to few-shot learning in computer vision",
# #             "Published paper on efficient attention mechanisms in ICLR 2024",
# #             "Teaching assistant for Advanced Deep Learning course",
# #             "Awarded departmental scholarship for outstanding academic performance"
# #         ],
# #         "courses": [
# #             "Advanced Deep Learning",
# #             "Natural Language Processing",
# #             "Computer Vision Systems",
# #             "Reinforcement Learning",
# #             "Big Data Analytics"
# #         ],
# #         "icon": "calendar"
# #     },
# #     {
# #         "degree": "BTech in Computer Science and Engineering",
# #         "institution": "Indian Institute of Technology, Indore",
# #         "period": "2016 - 2022",
# #         "description": "Gained strong foundations in computer science principles, algorithms, data structures, and software engineering practices with a focus on AI applications and system design.",
# #         "achievements": [
# #             "Graduated with honors, top 5% of class",
# #             "Led research project on neural network optimization",
# #             "Internship at Microsoft Research on reinforcement learning",
# #             "Received Best Undergraduate Thesis Award",
# #             "Published 2 papers in top-tier conferences"
# #         ],
# #         "courses": [
# #             "Data Structures and Algorithms",
# #             "Machine Learning Fundamentals",
# #             "Operating Systems",
# #             "Database Management Systems",
# #             "Software Engineering"
# #         ],
# #         "icon": "award"
# #     },
# #     {
# #         "degree": "AI Research Certificate Program",
# #         "institution": "Stanford University (Online)",
# #         "period": "2023",
# #         "description": "Intensive six-month program focused on cutting-edge AI research methodologies, advanced neural network architectures, and practical implementation of research papers.",
# #         "achievements": [
# #             "Completed with distinction (98% score)",
# #             "Implemented 5 state-of-the-art models from recent research papers",
# #             "Final project selected for showcase in program highlights",
# #             "Mentored by leading researchers in the field"
# #         ],
# #         "courses": [
# #             "Research Methods in AI",
# #             "Neural Network Architecture Design",
# #             "Paper Implementation Practicum",
# #             "AI Ethics and Responsibility"
# #         ],
# #         "icon": "book-open"
# #     }
# # ]
# # @app.route('/')
# # def index():
# #     # The index page automatically uses the full 'projects' list
# #     return render_template('index.html',
# #                           skills=skills,
# #                           additional_skills=additional_skills,
# #                           projects=projects, # Passed to the template
# #                           education=education,
# #                           character_profile=character_profile)

# # @app.route('/projects/<project_id>')
# # def project_detail(project_id):
# #     current_project = None
# #     current_index = -1

# #     # Find the current project and its index in the list
# #     for i, p in enumerate(projects):
# #         if p["id"] == project_id:
# #             current_project = p
# #             current_index = i
# #             break

# #     if current_project is None:
# #         abort(404) # Or redirect to index: return redirect(url_for('index'))

# #     # Determine previous and next project indices (with wrap-around)
# #     num_projects = len(projects)
# #     prev_index = (current_index - 1 + num_projects) % num_projects
# #     next_index = (current_index + 1) % num_projects

# #     # Get the previous and next project dictionaries
# #     prev_project = projects[prev_index]
# #     next_project = projects[next_index]

# #     # Render the specific template for the project, passing navigation data
# #     template_name = f'projects/{project_id}.html'
# #     try:
# #         return render_template(template_name,
# #                                project=current_project,
# #                                prev_project=prev_project, # Pass the whole project dict
# #                                next_project=next_project) # Pass the whole project dict
# #     except Exception as e:
# #         # Fallback if the specific template doesn't exist (optional)
# #         print(f"Error rendering template {template_name}: {e}")
# #         abort(404) # Or handle differently

# # @app.route('/toggle-theme')
# # def toggle_theme():
# #     # This would normally handle server-side theme preference saving
# #     # But for now, it's all done client-side with JavaScript
# #     return '', 204


# # if __name__ == '__main__':
# #     app.run(debug=True)


# # portfolio_website/app.py

# from flask import Flask, render_template, redirect, url_for, abort # Added abort
# import math

# app = Flask(__name__)

# # Make math functions available to templates
# # These might not be strictly necessary anymore if not used in templates
# @app.template_filter('cos')
# def cos_filter(value):
#     return math.cos(value)

# @app.template_filter('sin')
# def sin_filter(value):
#     return math.sin(value)

# # --- Data for the portfolio ---
# # Define this data *before* the routes that use it.

# skills = [
#     {"name": "Machine Learning", "icon": "brain", "level": 90, "category": "Expert"},
#     {"name": "Natural Language Processing", "icon": "message-circle", "level": 85, "category": "Expert"}, # Changed icon
#     {"name": "Deep Learning", "icon": "cpu", "level": 88, "category": "Expert"},
#     {"name": "Data Analysis", "icon": "database", "level": 92, "category": "Expert"},
#     {"name": "Python", "icon": "code", "level": 95, "category": "Expert"},
#     {"name": "TensorFlow/PyTorch", "icon": "layers", "level": 87, "category": "Expert"} # Changed icon
# ]

# additional_skills = [
#     'SQL', 'NoSQL', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'Git', 'CI/CD',
#     'RESTful APIs', 'Flask', 'FastAPI', 'React', 'Data Visualization',
#     'Scikit-learn', 'Pandas', 'NumPy', 'BERT', 'GPT', 'Computer Vision'
# ]

# # IMPORTANT: This list now defines the order for project navigation.
# # Add new projects to this list. Ensure 'id' is unique and matches the template filename.
# projects = [
#     {
#         "id": "sentiment-analysis", # Should be unique
#         "title": "Sentiment Analysis Engine",
#         "description": "Developed an advanced NLP model for multi-lingual sentiment analysis with 94% accuracy across 7 languages.",
#         "tags": ["NLP", "PyTorch", "Transformers"],
#         "thumbnail": "sentiment-analysis.jpg", # Ensure image exists in static/images/projects/
#         "github": "https://github.com/kee1bo/sentiment-analysis", # Corrected username
#         "medium": None, # Example: "https://medium.com/@neerudisaivikas/sentiment-analysis",
#         "youtube": None, # Example: "https://youtube.com/watch?v=...",
#         "demo": None # Example: "https://demo.sentiment-analysis.com"
#     },
#     {
#         "id": "computer-vision", # Unique ID
#         "title": "Computer Vision System",
#         "description": "Built an object detection and tracking system for retail analytics, reducing inventory errors by 35%.",
#         "tags": ["Computer Vision", "TensorFlow", "OpenCV"],
#         "thumbnail": "computer-vision.jpg", # Ensure image exists
#         "github": "https://github.com/kee1bo/computer-vision", # Corrected username
#         "medium": None,
#         "youtube": None,
#         "demo": None
#     },
#     {
#         "id": "predictive-analytics", # Unique ID
#         "title": "Predictive Analytics Platform",
#         "description": "Created a time-series forecasting solution for supply chain optimization, improving efficiency by 28%.",
#         "tags": ["Time Series", "LSTM", "Prophet"],
#         "thumbnail": "predictive-analytics.jpg", # Ensure image exists
#         "github": "https://github.com/kee1bo/predictive-analytics", # Corrected username
#         "medium": None,
#         "youtube": None,
#         "demo": None
#     },
#     {
#         "id": "permanent-vs-determinant", # Unique ID
#         "title": "Permanent vs Determinant",
#         "description": "An exploratory study on matrix computations with applications in quantum computing and graph theory.",
#         "tags": ["Linear Algebra", "Quantum Computing", "Graph Theory"],
#         "thumbnail": "permanent-determinant.jpg", # Ensure image exists
#         "github": "https://github.com/kee1bo/permanent-vs-determinant", # Corrected username
#         "medium": None,
#         "youtube": None,
#         "demo": None
#     },
#     # Add this dictionary to the 'projects' list in app.py
#     {
#         "id": "rna-folding", # Used in URL and for template filename
#         "title": "mRNA Vaccine Degradation Prediction",
#         "description": "Developed deep learning models to predict mRNA degradation rates for the Stanford OpenVaccine Kaggle competition.",
#         "tags": ["Bioinformatics", "Deep Learning", "Kaggle", "RNA", "Regression"],
#         "thumbnail": "rna-folding.jpg", # Create a placeholder or actual image
#         # Add your actual links or leave as None
#         "github": "https://github.com/kee1bo/kaggle-openvaccine", # Example link
#         "medium": None,
#         "youtube": None,
#         "demo": "https://www.kaggle.com/c/stanford-covid-vaccine", # Link to competition
#     }
#     # Make sure there's a comma after the closing brace if it's not the last item
#     # *** Add your new project dictionary here following the same structure ***
#     # Example:
#     # {
#     #     "id": "new-project-id", # MUST match the filename 'new-project-id.html'
#     #     "title": "My Awesome New Project",
#     #     "description": "Description of the new project.",
#     #     "tags": ["Tag1", "Tag2"],
#     #     "thumbnail": "new-project.jpg", # Ensure image exists
#     #     "github": "https://github.com/kee1bo/new-project-repo",
#     #     "medium": None,
#     #     "youtube": None,
#     #     "demo": None
#     # },
# ]

# education = [
#      {
#         "degree": "MTech in Artificial Intelligence and Data Science",
#         "institution": "JNTUHCE Sulthanpur",
#         "period": "2024 - 2026",
#         "description": "Specializing in advanced machine learning algorithms, deep learning architectures, and large-scale data processing techniques with a focus on practical applications in industry and research.",
#         "achievements": [
#             "Ongoing research on transformer architectures for multimodal learning",
#             "Working on a novel approach to few-shot learning in computer vision",
#             "Published paper on efficient attention mechanisms in ICLR 2024",
#             "Teaching assistant for Advanced Deep Learning course",
#             "Awarded departmental scholarship for outstanding academic performance"
#         ],
#         "courses": [
#             "Advanced Deep Learning",
#             "Natural Language Processing",
#             "Computer Vision Systems",
#             "Reinforcement Learning",
#             "Big Data Analytics"
#         ],
#         "icon": "graduation-cap" # Example Lucide icon name
#     },
#     {
#         "degree": "BTech in Computer Science and Engineering",
#         "institution": "Indian Institute of Technology, Indore",
#         "period": "2016 - 2022",
#         "description": "Gained strong foundations in computer science principles, algorithms, data structures, and software engineering practices with a focus on AI applications and system design.",
#         "achievements": [
#             "Graduated with honors, top 5% of class",
#             "Led research project on neural network optimization",
#             "Internship at Microsoft Research on reinforcement learning",
#             "Received Best Undergraduate Thesis Award",
#             "Published 2 papers in top-tier conferences"
#         ],
#         "courses": [
#             "Data Structures and Algorithms",
#             "Machine Learning Fundamentals",
#             "Operating Systems",
#             "Database Management Systems",
#             "Software Engineering"
#         ],
#         "icon": "award" # Example Lucide icon name
#     },
#     {
#         "degree": "AI Research Certificate Program",
#         "institution": "Stanford University (Online)",
#         "period": "2023",
#         "description": "Intensive six-month program focused on cutting-edge AI research methodologies, advanced neural network architectures, and practical implementation of research papers.",
#         "achievements": [
#             "Completed with distinction (98% score)",
#             "Implemented 5 state-of-the-art models from recent research papers",
#             "Final project selected for showcase in program highlights",
#             "Mentored by leading researchers in the field"
#         ],
#         "courses": [
#             "Research Methods in AI",
#             "Neural Network Architecture Design",
#             "Paper Implementation Practicum",
#             "AI Ethics and Responsibility"
#         ],
#         "icon": "book-open" # Example Lucide icon name
#     }
# ]

# character_profile = {
#     "name": "AI Engineer",
#     "level": 32,
#     "class": "Data Scientist",
#     "stats": {
#         "intelligence": 95,
#         "creativity": 88,
#         "speed": 82,
#         "persistence": 90,
#         "collaboration": 85
#     },
#     "special_abilities": [
#         "Pattern Recognition",
#         "Algorithm Mastery",
#         "Data Intuition",
#         "Code Optimization"
#     ]
# }

# # --- Flask Routes ---

# @app.route('/')
# def index():
#     """Renders the main index page with all projects."""
#     return render_template('index.html',
#                           skills=skills,
#                           additional_skills=additional_skills,
#                           projects=projects, # Pass the full list
#                           education=education,
#                           character_profile=character_profile)

# @app.route('/projects/<project_id>')
# def project_detail(project_id):
#     """Renders the detail page for a specific project."""
#     current_project = None
#     current_index = -1

#     # Find the current project and its index in the list
#     for i, p in enumerate(projects):
#         if p["id"] == project_id:
#             current_project = p
#             current_index = i
#             break

#     # If project ID is not found in our list, show a 404 error
#     if current_project is None:
#         print(f"Project with ID '{project_id}' not found.")
#         abort(404)

#     # Determine previous and next project dictionaries based on list order
#     num_projects = len(projects)
#     # Use modulo arithmetic for wrap-around navigation
#     prev_index = (current_index - 1 + num_projects) % num_projects
#     next_index = (current_index + 1) % num_projects

#     prev_project = projects[prev_index]
#     next_project = projects[next_index]

#     # Construct the template filename based on the project ID
#     template_name = f'projects/{project_id}.html'

#     try:
#         # Render the specific template, passing current, previous, and next project data
#         return render_template(template_name,
#                                project=current_project,
#                                prev_project=prev_project, # Pass the whole dict
#                                next_project=next_project) # Pass the whole dict
#     except Exception as e:
#         # If the template file doesn't exist or another error occurs
#         print(f"Error rendering template {template_name}: {e}")
#         abort(404) # Show 404 if template is missing

# @app.route('/toggle-theme')
# def toggle_theme():
#     """Placeholder for potential server-side theme logic (currently client-side)."""
#     # Client-side JavaScript handles theme toggling via localStorage
#     return '', 204 # No content response

# # --- Error Handlers (Optional but Recommended) ---
# @app.errorhandler(404)
# def page_not_found(e):
#     """Custom 404 error page."""
#     # You can create a custom 404.html template
#     # return render_template('404.html'), 404
#     return "Page Not Found", 404 # Simple text response

# @app.errorhandler(500)
# def internal_server_error(e):
#     """Custom 500 error page."""
#     # return render_template('500.html'), 500
#     return "Internal Server Error", 500

# # --- Run the App ---
# if __name__ == '__main__':
#     # Debug mode automatically reloads when code changes
#     # Set debug=False for production environments
#     app.run(debug=True)


# portfolio_website/app.py

from flask import Flask, render_template, redirect, url_for, abort
import math

app = Flask(__name__)

# --- Data for the portfolio ---

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

# Removed character_profile dictionary

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main index page."""
    return render_template('index.html',
                          skills=skills,
                          additional_skills=additional_skills,
                          projects=projects,
                          education=education) # Removed character_profile

@app.route('/projects/<project_id>')
def project_detail(project_id):
    """Renders the detail page for a specific project."""
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

@app.route('/toggle-theme')
def toggle_theme():
    """Placeholder for server-side theme logic."""
    return '', 204

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page."""
    return "Page Not Found", 404

@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 error page."""
    return "Internal Server Error", 500
@app.route('/toggle-theme')
def toggle_theme():
    """Placeholder for server-side theme logic."""
    return '', 204
# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)