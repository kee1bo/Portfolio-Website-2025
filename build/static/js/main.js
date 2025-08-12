// portfolio_website/static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    // Initialize navigation (handles clicks and initial load scroll)
    initNavigation();

    // Initialize mobile menu
    initMobileMenu();

    // Initialize scroll-triggered animations (like skill bars)
    initScrollAnimations(); // Assuming skills.js handles its own logic if present

    // Setup contact form submission handling
    setupContactForm();

    // Add fade-in animations for page sections on scroll
    addFadeInAnimations();

    // Add scrolled class to header
    initHeaderScroll();


    /**
     * Function to smoothly scroll to an element
     * @param {string} selector - CSS selector for the target element
     * @param {number} offset - Pixels to offset from the top (e.g., header height)
     */
    function smoothScrollTo(selector, offset = 80) {
        const element = document.querySelector(selector);
        if (element) {
            const currentHeaderHeight = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-height'), 10) || offset;
            const elementTop = element.getBoundingClientRect().top + window.pageYOffset;
            window.scrollTo({
                top: elementTop - currentHeaderHeight,
                behavior: 'smooth'
            });
        } else {
            console.warn("Smooth scroll target not found:", selector);
        }
    }

    /**
     * Initialize navigation highlighting and click handling.
     */
    function initNavigation() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        const headerHeight = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-height'), 10) || 70;
        const isIndexPage = !!document.getElementById('hero'); // More robust check

        if (isIndexPage && window.location.hash) {
            const hash = window.location.hash;
            if (document.querySelector(hash)) { // Check if element exists
                setTimeout(() => {
                    smoothScrollTo(hash, headerHeight);
                }, 100);
            }
        }

        let activeSectionId = '';
        let scrollTimeout;
        function updateActiveLink() {
             if (!isIndexPage || sections.length === 0) return;

             clearTimeout(scrollTimeout);
             scrollTimeout = setTimeout(() => {
                  const scrollPosition = window.scrollY + headerHeight + 50; // Adjusted offset

                  let foundSection = false;
                  sections.forEach(section => {
                      if (section) {
                          const sectionTop = section.offsetTop;
                          const sectionHeight = section.offsetHeight;
                          if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                              activeSectionId = section.getAttribute('id');
                              foundSection = true;
                          }
                      }
                  });

                  if (!foundSection) {
                      if (window.scrollY < (sections[0]?.offsetTop - headerHeight * 1.5)) {
                           activeSectionId = sections[0]?.getAttribute('id') || (isIndexPage ? 'hero' : '');
                      } else if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 150) { // Near bottom with buffer
                           activeSectionId = sections[sections.length - 1]?.getAttribute('id');
                      }
                  }

                  navLinks.forEach(link => {
                      if (link && typeof link.getAttribute === 'function') {
                          const linkHref = link.getAttribute('href');
                          const sectionIdFromHref = linkHref?.includes('#') ? linkHref.substring(linkHref.indexOf('#') + 1) : null;
                          const sectionIdFromData = link.getAttribute('data-section');
                          const targetSectionId = sectionIdFromData || sectionIdFromHref;

                          if (targetSectionId === activeSectionId) {
                              link.classList.add('active');
                          } else {
                              link.classList.remove('active');
                          }
                      }
                  });
             }, 50);
        }

        if (isIndexPage) {
            window.addEventListener('scroll', updateActiveLink, { passive: true });
            updateActiveLink();
        }

        navLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                const targetId = href?.includes('#') ? href.substring(href.indexOf('#')) : null;

                if (targetId) {
                    const targetElement = document.querySelector(targetId);
                    if ((isIndexPage || href.startsWith('#')) && targetElement) {
                        e.preventDefault();
                        smoothScrollTo(targetId, headerHeight);

                         const mobileNav = document.getElementById('mobile-nav');
                         const mobileToggle = document.getElementById('mobile-menu-toggle');
                         if (mobileNav && mobileNav.classList.contains('open')) {
                             mobileNav.classList.remove('open');
                             mobileToggle?.setAttribute('aria-expanded', 'false');
                             mobileToggle?.querySelector('.icon-menu')?.classList.remove('hidden');
                             mobileToggle?.querySelector('.icon-close')?.classList.add('hidden');
                         }
                    }
                }
            });
        });
    }

    function initMobileMenu() {
        const mobileToggle = document.getElementById('mobile-menu-toggle');
        const mobileNav = document.getElementById('mobile-nav');
        const menuIcon = mobileToggle?.querySelector('.icon-menu');
        const closeIcon = mobileToggle?.querySelector('.icon-close');

        if (!mobileToggle || !mobileNav || !menuIcon || !closeIcon) {
            console.warn("Mobile menu elements not found.");
            return;
        }

        mobileToggle.addEventListener('click', function() {
            const isOpen = mobileNav.classList.toggle('open');
            menuIcon.classList.toggle('hidden');
            closeIcon.classList.toggle('hidden');
            this.setAttribute('aria-expanded', String(isOpen));
            document.body.classList.toggle('no-scroll', isOpen);
        });
    }

    function initScrollAnimations() { /* Placeholder */ }

    function setupContactForm() {
      const contactForm = document.getElementById('contactForm');
      if (!contactForm) return;
      contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        // ... your form submission logic ...
        alert('Form submitted (simulated)!');
        contactForm.reset();
      });
    }

    function addFadeInAnimations() {
      const animateElements = document.querySelectorAll('section > .container, .about-content > div, .skills-grid > .skill-card, .projects-grid > .project-card, .contact-content > div, .timeline-item');
      if (animateElements.length === 0) return;
      // ... (rest of fade-in animation logic is fine) ...
      const styleId = 'fade-in-styles-dynamic';
      if (!document.getElementById(styleId)) {
          const style = document.createElement('style');
          style.id = styleId;
          style.textContent = `
            .fade-in-prepare {
              opacity: 0;
              transform: translateY(25px);
              transition: opacity 0.7s var(--transition-easing), transform 0.7s var(--transition-easing);
            }
            .fade-in-visible {
              opacity: 1;
              transform: translateY(0);
            }
          `;
          document.head.appendChild(style);
      }
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
          if (entry.isIntersecting) {
            entry.target.style.transitionDelay = `${index * 50}ms`;
            entry.target.classList.add('fade-in-visible');
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });
      animateElements.forEach(element => {
        element.classList.add('fade-in-prepare');
        observer.observe(element);
      });
    }

    function initHeaderScroll() {
        const header = document.getElementById('pageHeader');
        if (!header) return;
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        }, { passive: true });
    }
});