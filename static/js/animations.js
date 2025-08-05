// Enhanced Animations for Web3-Inspired Portfolio

// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

// Animate elements on scroll
const animateOnScroll = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
        }
    });
}, observerOptions);

// Initialize scroll animations
document.addEventListener('DOMContentLoaded', () => {
    // Add animation classes to elements
    const animatedElements = document.querySelectorAll('.skill-card, .project-card, .highlight-item, .timeline-item');
    animatedElements.forEach(el => {
        el.classList.add('animate-on-scroll');
        animateOnScroll.observe(el);
    });

    // Initialize parallax effects
    initParallax();
    
    // Initialize floating elements
    initFloatingElements();
    
    // Initialize particle effects
    initParticleEffects();
});

// Parallax effect for hero section
function initParallax() {
    const heroSection = document.querySelector('.hero-section');
    if (!heroSection) return;

    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        const animatedBg = document.querySelector('.animated-bg');
        if (animatedBg) {
            animatedBg.style.transform = `translateY(${rate}px)`;
        }
    });
}

// Floating animation for elements
function initFloatingElements() {
    const floatingElements = document.querySelectorAll('.profile-back, .skill-icon, .project-card-icon');
    
    floatingElements.forEach((el, index) => {
        el.style.animationDelay = `${index * 0.2}s`;
        el.classList.add('floating');
    });
}

// Particle effects for background
function initParticleEffects() {
    const heroSection = document.querySelector('.hero-section');
    if (!heroSection) return;

    // Create particle container
    const particleContainer = document.createElement('div');
    particleContainer.className = 'particle-container';
    heroSection.appendChild(particleContainer);

    // Generate particles
    for (let i = 0; i < 20; i++) {
        createParticle(particleContainer);
    }
}

function createParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Random properties
    const size = Math.random() * 4 + 2;
    const x = Math.random() * 100;
    const y = Math.random() * 100;
    const duration = Math.random() * 20 + 10;
    const delay = Math.random() * 10;
    
    particle.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${x}%;
        top: ${y}%;
        animation-duration: ${duration}s;
        animation-delay: ${delay}s;
    `;
    
    container.appendChild(particle);
}

// Smooth scroll for navigation links
document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const headerHeight = document.querySelector('.header').offsetHeight;
                const targetPosition = targetSection.offsetTop - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
});

// Active navigation highlighting
function updateActiveNav() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.clientHeight;
        
        if (window.pageYOffset >= sectionTop && window.pageYOffset < sectionTop + sectionHeight) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `/#${current}`) {
            link.classList.add('active');
        }
    });
}

// Header scroll effect
function initHeaderScroll() {
    const header = document.querySelector('.header');
    
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 100) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
        
        updateActiveNav();
    });
}

// Initialize header scroll effect
document.addEventListener('DOMContentLoaded', initHeaderScroll);

// Skill progress animation
function animateSkillProgress() {
    const progressBars = document.querySelectorAll('.progress-fill');
    
    const progressObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target;
                const level = progressBar.getAttribute('data-level');
                progressBar.style.width = `${level}%`;
            }
        });
    }, { threshold: 0.5 });
    
    progressBars.forEach(bar => {
        progressObserver.observe(bar);
    });
}

// Initialize skill progress animation
document.addEventListener('DOMContentLoaded', animateSkillProgress);

// Typing effect for hero title
function initTypingEffect() {
    const heroTitle = document.querySelector('.hero-title');
    if (!heroTitle) return;
    
    const text = heroTitle.textContent;
    heroTitle.textContent = '';
    
    let i = 0;
    const typeWriter = () => {
        if (i < text.length) {
            heroTitle.textContent += text.charAt(i);
            i++;
            setTimeout(typeWriter, 100);
        }
    };
    
    // Start typing effect after a delay
    setTimeout(typeWriter, 1000);
}

// Initialize typing effect
document.addEventListener('DOMContentLoaded', initTypingEffect);

// Mouse trail effect
function initMouseTrail() {
    const trail = document.createElement('div');
    trail.className = 'mouse-trail';
    document.body.appendChild(trail);
    
    let mouseX = 0;
    let mouseY = 0;
    let trailX = 0;
    let trailY = 0;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
    
    function animateTrail() {
        trailX += (mouseX - trailX) * 0.1;
        trailY += (mouseY - trailY) * 0.1;
        
        trail.style.left = trailX + 'px';
        trail.style.top = trailY + 'px';
        
        requestAnimationFrame(animateTrail);
    }
    
    animateTrail();
}

// Initialize mouse trail (optional - can be disabled for performance)
// document.addEventListener('DOMContentLoaded', initMouseTrail);

// Export functions for use in other scripts
window.portfolioAnimations = {
    animateOnScroll,
    initParallax,
    initFloatingElements,
    initParticleEffects,
    updateActiveNav,
    animateSkillProgress
};