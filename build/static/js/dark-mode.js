// Dark Mode Toggle for Web3-Inspired Portfolio

// Theme state management
let currentTheme = localStorage.getItem('theme') || 'light';

// DOM elements
const themeToggle = document.getElementById('theme-toggle');
const body = document.body;
const sunIcon = themeToggle?.querySelector('.sun');
const moonIcon = themeToggle?.querySelector('.moon');

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', () => {
    applyTheme(currentTheme);
    updateThemeUI();
});

// Theme toggle click handler
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        currentTheme = currentTheme === 'light' ? 'dark' : 'light';
        applyTheme(currentTheme);
        updateThemeUI();
        localStorage.setItem('theme', currentTheme);
    });
}

// Apply theme to document
function applyTheme(theme) {
    if (theme === 'dark') {
        body.setAttribute('data-theme', 'dark');
        body.classList.remove('theme-light');
        body.classList.add('theme-dark');
    } else {
        body.setAttribute('data-theme', 'light');
        body.classList.remove('theme-dark');
        body.classList.add('theme-light');
    }
}

// Update theme toggle UI
function updateThemeUI() {
    if (!themeToggle) return;

    if (currentTheme === 'dark') {
        // Show sun icon for switching to light mode
        if (sunIcon) sunIcon.classList.remove('hidden');
        if (moonIcon) moonIcon.classList.add('hidden');
        themeToggle.setAttribute('aria-label', 'Switch to light mode');
    } else {
        // Show moon icon for switching to dark mode
        if (sunIcon) sunIcon.classList.add('hidden');
        if (moonIcon) moonIcon.classList.remove('hidden');
        themeToggle.setAttribute('aria-label', 'Switch to dark mode');
    }
}

// Check for system preference
function checkSystemPreference() {
    if (localStorage.getItem('theme')) return; // User has made a choice

    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    currentTheme = prefersDark ? 'dark' : 'light';
    applyTheme(currentTheme);
    updateThemeUI();
}

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) { // Only auto-switch if user hasn't made a choice
        currentTheme = e.matches ? 'dark' : 'light';
        applyTheme(currentTheme);
        updateThemeUI();
    }
});

// Initialize system preference check
checkSystemPreference();

// Export for use in other scripts
window.themeManager = {
    getCurrentTheme: () => currentTheme,
    setTheme: (theme) => {
        currentTheme = theme;
        applyTheme(theme);
        updateThemeUI();
        localStorage.setItem('theme', theme);
    },
    toggleTheme: () => {
        currentTheme = currentTheme === 'light' ? 'dark' : 'light';
        applyTheme(currentTheme);
        updateThemeUI();
        localStorage.setItem('theme', currentTheme);
    }
};