document.addEventListener('DOMContentLoaded', function() {
    // Initialize animations for project details page
    initProjectDetailAnimations();

    // Add interactive effects to tech tags
    initTechTagEffects();

    // Add scroll animations for project sections
    initScrollAnimations();

    /**
     * Initialize animations for various elements on the project detail page
     */
    function initProjectDetailAnimations() {
      // Animate algorithm and application cards
      const cards = document.querySelectorAll('.algorithm-card, .application-card');

      cards.forEach((card, index) => {
        // Create a slight delay for each card
        card.style.opacity = 0;
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';

        setTimeout(() => {
          card.style.opacity = 1;
          card.style.transform = 'translateY(0)';
        }, 300 + (index * 100));

        // Add hover interactions
        card.addEventListener('mouseenter', function() {
          const icon = this.querySelector('.algorithm-icon, .application-icon');
          if (icon) {
            icon.style.transform = 'scale(1.1)';
          }
        });

        card.addEventListener('mouseleave', function() {
          const icon = this.querySelector('.algorithm-icon, .application-icon');
          if (icon) {
            icon.style.transform = '';
          }
        });
      });

      // Animate theorem cards
      const theoremCards = document.querySelectorAll('.theorem-card');

      theoremCards.forEach((card, index) => {
        card.style.opacity = 0;
        card.style.transform = 'translateX(-20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';

        setTimeout(() => {
          card.style.opacity = 1;
          card.style.transform = 'translateX(0)';
        }, 500 + (index * 150));
      });

      // Animate math formulas
      const mathFormulas = document.querySelectorAll('.math-formula');

      mathFormulas.forEach((formula, index) => {
        formula.style.opacity = 0;
        formula.style.transition = 'opacity 1s ease';

        setTimeout(() => {
          formula.style.opacity = 1;
        }, 800 + (index * 200));
      });

      // Animate visualization containers
      const visualizations = document.querySelectorAll('.visualization-container');

      visualizations.forEach((viz, index) => {
        viz.style.opacity = 0;
        viz.style.transform = 'scale(0.95)';
        viz.style.transition = 'opacity 0.8s ease, transform 0.8s ease';

        setTimeout(() => {
          viz.style.opacity = 1;
          viz.style.transform = 'scale(1)';
        }, 400 + (index * 300));
      });
    }

    /**
     * Add interactive sparkle effects to tech tags
     */
    function initTechTagEffects() {
      const techTags = document.querySelectorAll('.tech-tag');

      techTags.forEach(tag => {
        tag.addEventListener('mousemove', function(e) {
          // Get cursor position relative to the element
          const rect = this.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / this.offsetWidth) * 100;
          const y = ((e.clientY - rect.top) / this.offsetHeight) * 100;

          // Update CSS variables to position the gradient
          this.style.setProperty('--x', `${x}%`);
          this.style.setProperty('--y', `${y}%`);
        });

        tag.addEventListener('mouseenter', function() {
          // Create sparkle effect on hover
          createSparkleEffect(this);
        });
      });
    }

    /**
     * Create sparkle effect on an element
     * @param {HTMLElement} element - The element to add sparkles to
     */
    function createSparkleEffect(element) {
      // Number of sparkles to create
      const sparkleCount = 4;

      // Create sparkles
      for (let i = 0; i < sparkleCount; i++) {
        const sparkle = document.createElement('span');
        sparkle.classList.add('sparkle');

        // Random position around the element
        const posX = -10 + Math.random() * 120; // -10% to 110%
        const posY = -10 + Math.random() * 120; // -10% to 110%

        // Random size
        const size = 3 + Math.random() * 8; // 3px to 11px

        // Random animation duration
        const duration = 0.4 + Math.random() * 0.8; // 0.4s to 1.2s

        // Style the sparkle
        sparkle.style.cssText = `
          position: absolute;
          left: ${posX}%;
          top: ${posY}%;
          width: ${size}px;
          height: ${size}px;
          background-color: var(--color-accent);
          border-radius: 50%;
          opacity: 0;
          pointer-events: none;
          z-index: 10;
          animation: sparkle ${duration}s ease-in-out forwards;
        `;

        // Add to element
        element.style.position = 'relative';
        element.appendChild(sparkle);

        // Remove after animation completes
        setTimeout(() => {
          if (sparkle.parentNode === element) {
            element.removeChild(sparkle);
          }
        }, duration * 1000);
      }
    }

    /**
     * Add scroll-based animations for project sections
     */
    function initScrollAnimations() {
      const sections = document.querySelectorAll('.project-section, .project-math, .project-tech');

      // Create an intersection observer for sections
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.1 });

      // Prepare sections for animation
      sections.forEach(section => {
        section.style.opacity = 0;
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.8s ease, transform 0.8s ease';

        // Add fade-in class when section becomes visible
        observer.observe(section);
      });

      // Add CSS rule for fade-in animation
      const style = document.createElement('style');
      style.textContent = `
        .fade-in {
          opacity: 1 !important;
          transform: translateY(0) !important;
        }
      `;
      document.head.appendChild(style);
    }
  });