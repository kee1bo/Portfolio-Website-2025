// portfolio_website/static/js/skills.js

document.addEventListener('DOMContentLoaded', function() {
  // Initialize skill cards interactions (progress bar animation)
  initSkillProgressAnimation();

  // Note: Functions related to the previous red sparkle effect
  // (initRedSparkleEffects, createParticleEffects, etc.) have been removed.

  /**
   * Initialize progress bar animation when skill card is scrolled into view.
   * This function targets the .progress-fill element inside the .skill-card.
   */
  function initSkillProgressAnimation() {
    // Select all skill card containers
    // Note: We observe the outer .skill-card div
    const skillCards = document.querySelectorAll('.skill-card');

    // Do nothing if no skill cards are found
    if (skillCards.length === 0) {
        console.warn("No elements with class 'skill-card' found for progress animation.");
        return;
    }

    // Create an Intersection Observer instance
    const observer = new IntersectionObserver((entries, observerInstance) => {
      // Loop through the entries (elements being observed)
      entries.forEach(entry => {
        // Check if the element is intersecting (visible in the viewport)
        if (entry.isIntersecting) {
          const card = entry.target; // The .skill-card element
          // Find the progress bar fill element within the current card
          const progressFill = card.querySelector('.progress-fill');

          // Check if the progress fill element exists
          if (progressFill) {
            // Get the skill level from the 'data-level' attribute
            const level = progressFill.getAttribute('data-level');
            // Validate the level attribute (must be between 0 and 100)
            if (level && !isNaN(level) && level >= 0 && level <= 100) {
              // Apply the transform to animate the width (scaleX)
              // The transition is defined in CSS
              progressFill.style.transform = `scaleX(${level / 100})`;

            } else {
              // Log a warning if the data-level attribute is missing or invalid
              console.warn("Invalid or missing data-level attribute for progress bar inside:", card);
              // Set a default state (e.g., 0 width) if level is invalid
              progressFill.style.transform = 'scaleX(0)';
            }
          } else {
            // Log a warning if the .progress-fill element is not found
            console.warn("Element with class '.progress-fill' not found inside:", card);
          }

          // Stop observing the current card once it has become visible and animated
          // This ensures the animation only runs once per page load
          observerInstance.unobserve(card);
        }
      });
    }, {
        threshold: 0.2 // Trigger animation when 20% of the skill card is visible
    });

    // Start observing each skill card element
    skillCards.forEach(card => {
      observer.observe(card);
    });
  } // End of initSkillProgressAnimation function

}); // End of DOMContentLoaded event listener
