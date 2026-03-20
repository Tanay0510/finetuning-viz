/**
 * Core Data Manager for llmviz.studio
 * Handles fetching global configuration and shared state.
 */

window.App = {
    config: null,
    
    async init() {
        try {
            const response = await fetch('/api/config');
            this.config = await response.json();
            
            // Set global shorthand for backward compatibility with existing scripts
            window.METHODS = this.config.methods;
            window.GPUS = this.config.gpus;
            window.MODELS = this.config.models;
            
            // Dispatch event so other scripts know data is ready
            document.dispatchEvent(new CustomEvent('configReady', { detail: this.config }));

            // Hide Global Loader
            setTimeout(() => {
                const loader = document.getElementById('global-loader');
                if (loader) {
                    loader.style.opacity = '0';
                    setTimeout(() => loader.remove(), 700);
                }
            }, 500);
        } catch (error) {
            console.error('Failed to load application configuration:', error);
        }
    }
};

// Start loading immediately
App.init();
