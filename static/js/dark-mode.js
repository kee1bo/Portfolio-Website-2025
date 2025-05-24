document.addEventListener('DOMContentLoaded', function() {
    // Get theme toggle button
    const themeToggle = document.getElementById('theme-toggle');
    
    // Initialize theme based on user preference or system preference
    initializeTheme();
    
    // Toggle theme when button is clicked
    themeToggle.addEventListener('click', toggleTheme);
    
    /**
     * Initialize theme based on local storage or system preference
     */
    function initializeTheme() {
      const savedTheme = localStorage.getItem('theme');
      
      if (savedTheme) {
        // Use saved theme preference
        document.body.className = savedTheme;
      } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // Use system dark mode preference if no saved preference
        document.body.className = 'theme-dark';
        localStorage.setItem('theme', 'theme-dark');
      } else {
        // Default to light theme
        document.body.className = 'theme-light';
        localStorage.setItem('theme', 'theme-light');
      }
      
      // Update UI to match current theme
      updateThemeUI();
    }
    
    /**
     * Toggle between light and dark themes
     */
    function toggleTheme() {
      if (document.body.classList.contains('theme-light')) {
        document.body.className = 'theme-dark';
        localStorage.setItem('theme', 'theme-dark');
      } else {
        document.body.className = 'theme-light';
        localStorage.setItem('theme', 'theme-light');
      }
      
      // Update UI to match new theme
      updateThemeUI();
    }
    
    /**
     * Update UI elements based on current theme
     */
    function updateThemeUI() {
      const isDarkTheme = document.body.classList.contains('theme-dark');
      
      // Set icon based on current theme
      const sunIcon = document.querySelector('.theme-toggle-icon.sun');
      const moonIcon = document.querySelector('.theme-toggle-icon.moon');
      
      if (isDarkTheme) {
        sunIcon.classList.add('hidden');
        moonIcon.classList.remove('hidden');
      } else {
        sunIcon.classList.remove('hidden');
        moonIcon.classList.add('hidden');
      }
      
      // Update lucide icons for the toggle button
      if (themeToggle.querySelector('[data-lucide="sun"]')) {
        themeToggle.querySelector('[data-lucide="sun"]').remove();
        themeToggle.querySelector('[data-lucide="moon"]').remove();
        lucide.createIcons({
          icons: {
            sun: sunIcon,
            moon: moonIcon
          }
        });
      } else {
        const sunElement = document.createElement('i');
        sunElement.dataset.lucide = 'sun';
        sunIcon.appendChild(sunElement);
        
        const moonElement = document.createElement('i');
        moonElement.dataset.lucide = 'moon';
        moonIcon.appendChild(moonElement);
        
        lucide.createIcons({
          icons: {
            sun: sunElement,
            moon: moonElement
          }
        });
      }
    }
  });