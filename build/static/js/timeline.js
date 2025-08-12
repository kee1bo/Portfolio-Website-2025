// portfolio_website/static/js/timeline.js

document.addEventListener('DOMContentLoaded', function() {
    const timeline = document.querySelector('.timeline');
    if (!timeline) return;

    /**
     * Initialize Intersection Observer for timeline items and line.
     */
    function initTimelineAnimations() {
        const timelineItems = timeline.querySelectorAll('.timeline-item');
        const timelineLine = timeline.querySelector('.timeline-line');

        const itemObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                    itemObserver.unobserve(entry.target); // Animate once
                }
            });
        }, { threshold: 0.15 }); // Trigger when 15% visible

        timelineItems.forEach((item, index) => {
            // Add a slight delay for staggered animation
            item.style.transitionDelay = `${index * 100}ms`;
            itemObserver.observe(item);
        });

        if (timelineLine) {
            const lineObserver = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) {
                    timeline.classList.add('loaded'); // Used to trigger line animation
                    lineObserver.unobserve(timelineLine);
                }
            }, { threshold: 0.1 });
            lineObserver.observe(timelineLine); // Observe the line itself or its container
        }
    }


    /**
     * Set up toggle buttons for achievements and courses.
     */
    function setupToggleButtons() {
        const toggleButtons = timeline.querySelectorAll('.timeline-toggle');

        if (toggleButtons.length === 0) return;

        toggleButtons.forEach(button => {
            button.addEventListener('click', function() {
                const targetType = this.getAttribute('data-target');
                const timelineItem = this.closest('.timeline-item');
                if (!timelineItem) return;

                const targetSection = timelineItem.querySelector(`.timeline-${targetType}`);
                if (!targetSection) {
                    console.warn(`Target section .timeline-${targetType} not found.`);
                    return;
                }

                const isExpanded = targetSection.classList.toggle('show');
                this.classList.toggle('expanded', isExpanded); // Add 'expanded' class to button

                // Update button text and ARIA attribute
                const buttonTextSpan = this.querySelector('span');
                const lucideIcon = this.querySelector('i[data-lucide]');

                if (isExpanded) {
                    if (buttonTextSpan) buttonTextSpan.textContent = `Hide ${targetType.charAt(0).toUpperCase() + targetType.slice(1)}`;
                    this.setAttribute('aria-expanded', 'true');
                    if (lucideIcon) lucideIcon.setAttribute('data-lucide', 'chevron-up');
                } else {
                    if (buttonTextSpan) buttonTextSpan.textContent = `Show ${targetType.charAt(0).toUpperCase() + targetType.slice(1)}`;
                    this.setAttribute('aria-expanded', 'false');
                    if (lucideIcon) lucideIcon.setAttribute('data-lucide', 'chevron-down');
                }

                // Re-initialize Lucide icon if it exists
                if (lucideIcon && typeof lucide !== 'undefined') {
                    lucide.createIcons({
                        elements: [lucideIcon] // Target only the changed icon
                    });
                }
            });

            // Set initial ARIA state
            const targetType = button.getAttribute('data-target');
            const timelineItem = button.closest('.timeline-item');
            if (timelineItem) {
                 const targetSection = timelineItem.querySelector(`.timeline-${targetType}`);
                 if (targetSection && targetSection.classList.contains('show')) {
                     button.setAttribute('aria-expanded', 'true');
                     button.classList.add('expanded');
                 } else {
                     button.setAttribute('aria-expanded', 'false');
                 }
            }
        });
    }

    initTimelineAnimations();
    setupToggleButtons();
});