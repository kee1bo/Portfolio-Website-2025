# Advanced AI Portfolio Website 🚀

A modern, feature-rich portfolio website showcasing AI, data science, and machine learning projects with advanced functionality including blogging, project management, and real-time analytics.

## ✨ Features

### 🎨 Modern Design
- **Liquid Glass Effects**: Advanced glassmorphism design with backdrop filters
- **Responsive Layout**: Optimized for all devices and screen sizes
- **Dark/Light Mode**: Automatic theme switching capabilities
- **Advanced Animations**: Smooth transitions, particle effects, and 3D transformations
- **Premium UI Components**: Custom-designed cards, buttons, and interactive elements

### 📝 Blog System
- **Rich Content Management**: Full-featured blog with Markdown support
- **Advanced Search & Filtering**: Search by content, filter by tags, pagination
- **Reading Time Estimation**: Automatic calculation of article reading time
- **View Tracking**: Analytics for blog post engagement
- **SEO Optimized**: Meta tags, structured data, and search engine optimization

### 💼 Project Showcase
- **Advanced Filtering**: Filter by technology, status, completion date
- **Interactive Project Cards**: Hover effects, metrics display, live demos
- **Detailed Project Pages**: Comprehensive project documentation
- **Technology Stack Visualization**: Visual representation of tech used
- **Performance Metrics**: Display of project KPIs and achievements

### 📊 Analytics & Insights
- **Real-time Analytics**: Track page views, user engagement
- **Performance Metrics**: Monitor website and project performance
- **Database Integration**: SQLite for content management and analytics
- **API Integration**: RESTful APIs for data access

### 🎥 Multimedia Integration
- **YouTube Integration**: Embedded video content with previews
- **GitHub Integration**: Automatic repository data fetching
- **Image Optimization**: Responsive images with lazy loading
- **Media Galleries**: Interactive project image galleries

### 🔧 Advanced Functionality
- **Database-Driven Content**: Dynamic content management
- **Contact Form**: Advanced contact form with validation
- **Resume Generation**: Downloadable PDF resume
- **Social Media Integration**: Links to professional profiles
- **Search Functionality**: Global search across all content

## 🛠 Technology Stack

### Backend
- **Flask**: Python web framework for backend logic
- **SQLite**: Lightweight database for content storage
- **Python Libraries**: 
  - `sqlite3` for database operations
  - `datetime` for time management
  - `math` for calculations
  - `werkzeug` for security features

### Frontend
- **HTML5**: Semantic markup structure
- **Advanced CSS3**: 
  - CSS Grid and Flexbox layouts
  - CSS Custom Properties (variables)
  - Advanced animations and transitions
  - Glassmorphism effects with backdrop-filter
- **Vanilla JavaScript**: 
  - ES6+ features
  - Intersection Observer API
  - Fetch API for asynchronous requests
  - Custom animation libraries

### Design System
- **Liquid Glass Theme**: Modern glassmorphism design language
- **Responsive Typography**: Fluid typography with clamp() functions
- **Color System**: Consistent color palette with CSS custom properties
- **Component Library**: Reusable UI components
- **Animation Framework**: Custom CSS animations and transitions

## 📁 Project Structure

```
Portfolio-Website-2025/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── populate_db.py                  # Database population script
├── build.py                        # Static site generation
├── portfolio.db                    # SQLite database (generated)
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── static/
│   ├── css/
│   │   ├── style.css              # Original styles
│   │   ├── advanced-styles.css    # Advanced liquid glass styles
│   │   ├── dark-mode.css          # Dark mode styles
│   │   └── project-details.css    # Project-specific styles
│   ├── js/
│   │   ├── main.js                # Main JavaScript functionality
│   │   ├── dark-mode.js           # Theme switching
│   │   ├── project-detail.js      # Project page interactions
│   │   └── advanced-effects.js    # Advanced animations
│   └── images/
│       ├── profile.jpg            # Profile image
│       └── project-thumbnails/    # Project images
├── templates/
│   ├── base.html                  # Base template
│   ├── index.html                 # Original homepage
│   ├── index-enhanced.html        # Enhanced homepage with advanced features
│   ├── blog/
│   │   ├── index.html            # Blog listing page
│   │   └── post.html             # Individual blog post
│   ├── projects/
│   │   ├── overview.html         # Projects overview with filtering
│   │   ├── sentiment-analysis.html
│   │   ├── computer-vision.html
│   │   ├── predictive-analytics.html
│   │   ├── permanent-vs-determinant.html
│   │   └── rna-folding.html
│   ├── partials/
│   │   ├── header.html           # Navigation header
│   │   ├── footer.html           # Site footer
│   │   └── navigation.html       # Navigation menu
│   └── errors/
│       ├── 404.html              # Custom 404 page
│       └── 500.html              # Custom error page
└── venv/                          # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Portfolio-Website-2025
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   python populate_db.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the website**
   Open your browser and navigate to `http://localhost:5000`

### Development Setup

For development with hot reloading:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## 📊 Database Schema

### Blog Posts
```sql
CREATE TABLE blog_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    reading_time INTEGER DEFAULT 5
);
```

### YouTube Videos
```sql
CREATE TABLE youtube_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    video_id TEXT UNIQUE NOT NULL,
    description TEXT,
    thumbnail_url TEXT,
    published_at TIMESTAMP,
    views INTEGER DEFAULT 0,
    duration TEXT,
    featured INTEGER DEFAULT 0
);
```

### GitHub Repositories
```sql
CREATE TABLE github_repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    html_url TEXT,
    language TEXT,
    stars INTEGER DEFAULT 0,
    forks INTEGER DEFAULT 0,
    updated_at TIMESTAMP,
    featured INTEGER DEFAULT 0
);
```

### Analytics
```sql
CREATE TABLE analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_path TEXT NOT NULL,
    visitor_ip TEXT,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎨 Design Features

### Liquid Glass Effects
The website implements advanced glassmorphism design with:
- **Backdrop Blur**: CSS `backdrop-filter: blur()` for glass-like transparency
- **Gradient Overlays**: Subtle color gradients for depth
- **Border Highlights**: Subtle border illumination on hover
- **Floating Elements**: Animated particles and floating UI elements

### Responsive Design
- **Mobile-First Approach**: Optimized for mobile devices
- **Flexible Grid Systems**: CSS Grid and Flexbox layouts
- **Fluid Typography**: Responsive font sizing with `clamp()`
- **Adaptive Components**: Components that adapt to screen size

### Advanced Animations
- **CSS Transitions**: Smooth property changes
- **Keyframe Animations**: Complex multi-step animations
- **Intersection Observer**: Trigger animations on scroll
- **Transform Effects**: 3D transformations and perspective

## 🔧 Customization

### Adding New Blog Posts
1. Use the admin interface (when implemented) or
2. Add directly to the database:
   ```python
   python -c "
   import sqlite3
   conn = sqlite3.connect('portfolio.db')
   c = conn.cursor()
   c.execute('INSERT INTO blog_posts (title, slug, content, published) VALUES (?, ?, ?, ?)', 
             ('Your Title', 'your-slug', 'Content here', 1))
   conn.commit()
   "
   ```

### Modifying Styles
- Edit `static/css/advanced-styles.css` for liquid glass effects
- Modify CSS custom properties in `:root` for color scheme changes
- Add new components following the existing design system

### Adding New Projects
Update the `projects` list in `app.py`:
```python
projects.append({
    "id": "new-project",
    "title": "Your Project Title",
    "description": "Project description",
    "tags": ["Technology", "Tags"],
    "thumbnail": "project-image.jpg",
    # ... other properties
})
```

## 📈 Analytics

The website includes built-in analytics tracking:
- **Page Views**: Track visits to each page
- **User Engagement**: Monitor time spent and interactions
- **Popular Content**: Identify most viewed projects and blogs
- **Geographic Data**: Track visitor locations (when enabled)

Access analytics at `/api/analytics` endpoint.

## 🔐 Security Features

- **Input Validation**: All user inputs are validated
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Content sanitization
- **CSRF Protection**: Form validation tokens
- **Rate Limiting**: API endpoint protection

## 🚀 Deployment

### Static Site Generation
Generate a static version for hosting:
```bash
python build.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### Heroku Deployment
1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: python app.py
   ```
3. Deploy:
   ```bash
   heroku create your-portfolio
   git push heroku main
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Design Inspiration**: Modern web design trends and glassmorphism
- **Libraries Used**: Flask, SQLite, and various CSS/JS libraries
- **Image Sources**: Unsplash for demo images
- **Icons**: Lucide React icon library

## 📞 Contact

- **Email**: neerudisaivikas@gmail.com
- **LinkedIn**: [Neerudi Sai Vikas](https://www.linkedin.com/in/neerudi-sai-vikas-721349226/)
- **GitHub**: [kee1bo](https://github.com/kee1bo)

---

**Built with ❤️ using Python, Flask, and modern web technologies.**