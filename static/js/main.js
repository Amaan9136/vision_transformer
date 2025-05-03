// DOM Content Loaded Event
document.addEventListener('DOMContentLoaded', function() {
    // Initialize animation effects
    initializeAnimations();
    
    // Initialize event listeners for image scraping
    initializeImageScraper();
    
    // Initialize drag and drop functionality
    initializeDragAndDrop();
    
    // Initialize tab switching
    initializeTabs();
    
    // Initialize search functionality
    initializeSearch();
    
    // Initialize number inputs
    initializeNumberInputs();
    
    // Initialize settings toggles
    initializeSettingsToggles();
    
    // Initialize "Find More Similar" functionality
    initializeMoreSimilarImages();
    
    // Initialize paste handling for base64 images
    initializePasteHandling();
    
    // Initialize web scraping button
    initializeWebScrapingButton();

    initializeWebSocketConnection();
    
    // Add logs link to header
    const header = document.querySelector('header');
    if (header) {
        const logsLink = document.createElement('a');
        logsLink.href = '/system_logs';
        logsLink.className = 'btn-secondary px-4 py-2 rounded-lg ml-4';
        logsLink.innerHTML = '<i class="fas fa-list-alt mr-2"></i> System Logs';
        
        // Find or create a container for the links
        let headerLinks = header.querySelector('.header-links');
        if (!headerLinks) {
            headerLinks = document.createElement('div');
            headerLinks.className = 'header-links flex items-center';
            header.appendChild(headerLinks);
        }
        
        headerLinks.appendChild(logsLink);
    }
});

// Initialize number inputs
function initializeNumberInputs() {
    const numberInputs = document.querySelectorAll('.number-input');
    
    numberInputs.forEach(container => {
        const input = container.querySelector('input[type="number"]');
        const incrementBtn = container.querySelector('.increment');
        const decrementBtn = container.querySelector('.decrement');
        
        if (input && incrementBtn && decrementBtn) {
            incrementBtn.addEventListener('click', () => {
                const currentValue = parseInt(input.value, 10);
                const max = parseInt(input.getAttribute('max'), 10);
                
                if (currentValue < max) {
                    input.value = currentValue + 1;
                    input.dispatchEvent(new Event('change'));
                }
            });
            
            decrementBtn.addEventListener('click', () => {
                const currentValue = parseInt(input.value, 10);
                const min = parseInt(input.getAttribute('min'), 10);
                
                if (currentValue > min) {
                    input.value = currentValue - 1;
                    input.dispatchEvent(new Event('change'));
                }
            });
            
            // Prevent non-numeric input
            input.addEventListener('input', () => {
                let value = parseInt(input.value, 10);
                const min = parseInt(input.getAttribute('min'), 10);
                const max = parseInt(input.getAttribute('max'), 10);
                
                if (isNaN(value)) {
                    input.value = min;
                } else {
                    value = Math.min(Math.max(value, min), max);
                    input.value = value;
                }
            });
        }
    });
    
    // Special handling for results page number input
    const resultsNumberInput = document.getElementById('num-images-results');
    if (resultsNumberInput) {
        resultsNumberInput.addEventListener('change', () => {
            // Get the current label
            const detectedLabel = document.getElementById('detected-label');
            if (detectedLabel && detectedLabel.textContent) {
                // Fetch more similar images with the new count
                fetchMoreSimilarImages(detectedLabel.textContent, parseInt(resultsNumberInput.value, 10));
            }
        });
    }
}

// Initialize settings toggles
function initializeSettingsToggles() {
    const uploadToggle = document.getElementById('upload-settings-toggle');
    const uploadSettings = document.getElementById('upload-advanced-settings');
    
    if (uploadToggle && uploadSettings) {
        uploadToggle.addEventListener('click', () => {
            uploadSettings.classList.toggle('hidden');
        });
    }
    
    const urlToggle = document.getElementById('url-settings-toggle');
    const urlSettings = document.getElementById('url-advanced-settings');
    
    if (urlToggle && urlSettings) {
        urlToggle.addEventListener('click', () => {
            urlSettings.classList.toggle('hidden');
        });
    }
}

// Initialize search functionality
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const clearSearchBtn = document.getElementById('clear-search');
    
    if (searchInput && searchBtn) {
        // Handle search button click
        searchBtn.addEventListener('click', () => {
            performSearch();
        });
        
        // Handle enter key press in search input
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
        
        // Handle clear search button click
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', () => {
                searchInput.value = '';
                hideSearchResults();
            });
        }
    }
}

// Perform search function
function performSearch() {
    const searchInput = document.getElementById('search-input');
    const searchTerm = searchInput.value.trim();
    
    if (!searchTerm) return;
    
    showLoading('Searching...');
    
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            term: searchTerm,
            numWebResults: 5 // Default number of web results to fetch
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            displaySearchResults(data);
        } else {
            showError(data.error || 'Search failed. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error searching:', error);
        hideLoading();
        showError('Search failed. Please try again.');
    });
}

// Display search results
function displaySearchResults(data) {
    const searchResults = document.getElementById('search-results');
    const searchContainer = document.getElementById('search-container');
    const noResults = document.getElementById('no-results');
    const resultsSection = document.getElementById('results');
    
    if (searchResults && searchContainer && noResults) {
        // Hide results section and show search results section
        if (resultsSection) resultsSection.classList.add('hidden');
        searchResults.classList.remove('hidden');
        
        // Clear previous results
        searchContainer.innerHTML = '';
        
        if (data.results && data.results.length > 0) {
            // Hide no results message
            noResults.classList.add('hidden');
            
            // Add each result to the container with animation delay
            data.results.forEach((result, index) => {
                const delay = index * 0.1;
                
                const card = document.createElement('div');
                card.className = 'image-card glass-panel p-4 rounded-lg overflow-hidden relative';
                card.style.animationDelay = `${delay}s`;
                
                // Add origin badge
                const originClass = result.origin === 'database' ? 'origin-database' : 'origin-web';
                const originText = result.origin === 'database' ? 'DB' : 'Web';
                
                const badge = document.createElement('div');
                badge.className = `image-origin-badge ${originClass}`;
                badge.textContent = originText;
                card.appendChild(badge);
                
                const img = document.createElement('img');
                img.src = result.source;
                img.alt = result.label;
                img.className = 'w-full h-40 object-cover rounded-lg mb-3 image-zoom';
                
                const label = document.createElement('p');
                label.className = 'text-gray-800 font-medium truncate';
                label.textContent = result.label;
                
                // Add timestamp if available
                if (result.timestamp) {
                    const timestamp = document.createElement('p');
                    timestamp.className = 'text-gray-500 text-xs';
                    
                    // Format timestamp
                    const date = new Date(result.timestamp);
                    timestamp.textContent = date.toLocaleString();
                    
                    card.appendChild(timestamp);
                }
                
                card.appendChild(img);
                card.appendChild(label);
                
                // Add click event to analyze this image
                card.addEventListener('click', () => {
                    analyzeImageUrl(result.source);
                });
                
                searchContainer.appendChild(card);
            });
        } else {
            // Show no results message
            noResults.classList.remove('hidden');
        }
        
        // Scroll to search results
        searchResults.scrollIntoView({ behavior: 'smooth' });
    }
}

// Hide search results
function hideSearchResults() {
    const searchResults = document.getElementById('search-results');
    const resultsSection = document.getElementById('results');
    
    if (searchResults) searchResults.classList.add('hidden');
    if (resultsSection) resultsSection.classList.remove('hidden');
}

// Initialize animations and visual effects
function initializeAnimations() {
    // Add particle background effect
    const particleContainer = document.createElement('div');
    particleContainer.id = 'particle-container';
    particleContainer.style.position = 'fixed';
    particleContainer.style.top = '0';
    particleContainer.style.left = '0';
    particleContainer.style.width = '100%';
    particleContainer.style.height = '100%';
    particleContainer.style.zIndex = '-1';
    particleContainer.style.overflow = 'hidden';
    document.body.appendChild(particleContainer);
    
    // Create particles
    for (let i = 0; i < 30; i++) {
        createParticle();
    }
    
    // Add smooth scroll to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add hover effect to cards
    const applyCardEffects = () => {
        document.querySelectorAll('.glass-panel').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
            });
        });
    };
    
    // Initialize card effects and set up a mutation observer to apply to new cards
    applyCardEffects();
    
    // Observe DOM changes to apply effects to dynamically added elements
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.addedNodes.length) {
                applyCardEffects();
            }
        });
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
}

// Create a floating particle effect
function createParticle() {
    const particle = document.createElement('div');
    const size = Math.random() * 10 + 5;
    
    particle.className = 'particle';
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.left = `${Math.random() * 100}vw`;
    particle.style.top = `${Math.random() * 100}vh`;
    particle.style.opacity = `${Math.random() * 0.3}`;
    
    document.getElementById('particle-container').appendChild(particle);
    
    // Animate particle movement
    animateParticle(particle);
}

// Animate particle floating effect
function animateParticle(particle) {
    const duration = Math.random() * 60 + 30; // seconds
    const xMove = Math.random() * 20 - 10; // pixels
    const yMove = Math.random() * 20 - 10; // pixels
    
    particle.style.transition = `transform ${duration}s linear, opacity 1s ease-in-out`;
    
    setTimeout(() => {
        particle.style.transform = `translate(${xMove}px, ${yMove}px)`;
    }, 100);
    
    // Reset after animation completes
    setTimeout(() => {
        particle.style.transform = 'none';
        particle.style.left = `${Math.random() * 100}vw`;
        particle.style.top = `${Math.random() * 100}vh`;
        
        // Recursive call for continuous animation
        animateParticle(particle);
    }, duration * 1000);
}

// Initialize image scraper functionality
function initializeImageScraper() {
    // Add URL input functionality
    const urlBtn = document.getElementById('url-btn');
    const urlInput = document.getElementById('url-input');
    const urlError = document.getElementById('url-error');
    
    if (urlBtn && urlInput) {
        urlBtn.addEventListener('click', () => {
            const imageUrl = urlInput.value.trim();
            if (imageUrl) {
                urlError.classList.add('hidden');
                
                // Get the number of images to find
                const numImagesInput = document.getElementById('num-images-url');
                const numImages = numImagesInput ? parseInt(numImagesInput.value, 10) : 5;
                
                // Get selected model
                const modelSelect = document.getElementById('url-insight-model');
                const model = modelSelect ? modelSelect.value : 'mistral:latest';
                
                // Get safe search setting
                const safeSearch = document.querySelector('input[name="url-safe-search"]:checked').value === 'on';
                
                analyzeImageUrl(imageUrl, numImages, model, safeSearch);
            } else {
                urlError.classList.remove('hidden');
            }
        });
        
        // Handle enter key in URL input
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const imageUrl = urlInput.value.trim();
                if (imageUrl) {
                    urlError.classList.add('hidden');
                    
                    // Get the number of images to find
                    const numImagesInput = document.getElementById('num-images-url');
                    const numImages = numImagesInput ? parseInt(numImagesInput.value, 10) : 5;
                    
                    // Get selected model
                    const modelSelect = document.getElementById('url-insight-model');
                    const model = modelSelect ? modelSelect.value : 'mistral:latest';
                    
                    // Get safe search setting
                    const safeSearch = document.querySelector('input[name="url-safe-search"]:checked').value === 'on';
                    
                    analyzeImageUrl(imageUrl, numImages, model, safeSearch);
                } else {
                    urlError.classList.remove('hidden');
                }
            }
        });
    }
}

// Initialize drag and drop functionality
function initializeDragAndDrop() {
    const dropArea = document.querySelector('.drop-area');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    if (dropArea && fileInput && browseBtn) {
        // Prevent default behaviors for drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area when file is dragged over
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.add('active');
            });
        });
        
        // Remove highlight when file is dragged out or dropped
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.remove('active');
            });
        });
        
        // Handle file drop
        dropArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                // Get the number of images to find
                const numImagesInput = document.getElementById('num-images-upload');
                const numImages = numImagesInput ? parseInt(numImagesInput.value, 10) : 5;
                
                // Get selected model
                const modelSelect = document.getElementById('insight-model');
                const model = modelSelect ? modelSelect.value : 'mistral:latest';
                
                // Get safe search setting
                const safeSearch = document.querySelector('input[name="safe-search"]:checked').value === 'on';
                
                handleImageFile(files[0], numImages, model, safeSearch);
            }
        });
        
        // Handle browse files click
        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                // Get the number of images to find
                const numImagesInput = document.getElementById('num-images-upload');
                const numImages = numImagesInput ? parseInt(numImagesInput.value, 10) : 5;
                
                // Get selected model
                const modelSelect = document.getElementById('insight-model');
                const model = modelSelect ? modelSelect.value : 'mistral:latest';
                
                // Get safe search setting
                const safeSearch = document.querySelector('input[name="safe-search"]:checked').value === 'on';
                
                handleImageFile(e.target.files[0], numImages, model, safeSearch);
            }
        });
    }
}

// Initialize tab switching
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Hide all tab panes
            tabPanes.forEach(pane => {
                pane.classList.add('hidden');
            });
            
            // Show the corresponding tab pane
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.remove('hidden');
        });
    });
}

// Initialize More Similar Images functionality
function initializeMoreSimilarImages() {
    const scrapeMoreBtn = document.getElementById('scrape-more-images');
    
    if (scrapeMoreBtn) {
        scrapeMoreBtn.addEventListener('click', () => {
            // Get the detected label
            const detectedLabel = document.getElementById('detected-label');
            
            if (detectedLabel && detectedLabel.textContent) {
                // Get the number of images to fetch
                const numImagesInput = document.getElementById('num-images-results');
                const numImages = numImagesInput ? parseInt(numImagesInput.value, 10) : 5;
                
                fetchMoreSimilarImages(detectedLabel.textContent, numImages);
            }
        });
    }
}

// Fetch more similar images
function fetchMoreSimilarImages(label, limit = 5) {
    if (!label) {
        showNotification('No object detected to find similar images for', 'error');
        return;
    }
    
    // Show loading indicator for similar images section
    const similarLoading = document.getElementById('similar-loading');
    const similarContainer = document.getElementById('similar-container');
    const noSimilar = document.getElementById('no-similar');
    
    if (similarLoading && similarContainer) {
        similarLoading.classList.remove('hidden');
        similarContainer.innerHTML = '';
        if (noSimilar) noSimilar.classList.add('hidden');
    }
    
    // Make the API request
    fetch('/scrape-similar', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            label: label,
            limit: limit
        })
    })
    .then(response => response.json())
    .then(data => {
        if (similarLoading) similarLoading.classList.add('hidden');
        
        if (data.success && data.images && data.images.length > 0) {
            // Update similar images section
            displaySimilarImages(data.images);
        } else {
            showNotification('No additional similar images found', 'info');
            if (noSimilar && similarContainer.children.length === 0) {
                noSimilar.classList.remove('hidden');
            }
        }
    })
    .catch(error => {
        console.error('Error fetching more similar images:', error);
        if (similarLoading) similarLoading.classList.add('hidden');
        showError('Failed to fetch additional similar images');
    });
}

// Display similar images
function displaySimilarImages(images) {
    const similarContainer = document.getElementById('similar-container');
    const noSimilar = document.getElementById('no-similar');
    
    if (!similarContainer) return;
    
    // Hide no results message
    if (noSimilar) noSimilar.classList.add('hidden');
    
    // Clear existing images
    similarContainer.innerHTML = '';
    
    // Add each image to the container with animation delay
    images.forEach((image, index) => {
        const delay = index * 0.1;
        
        const card = document.createElement('div');
        card.className = 'image-card glass-panel p-4 rounded-lg overflow-hidden relative';
        card.style.animationDelay = `${delay}s`;
        
        // Add origin badge if available
        if (image.origin) {
            const originClass = image.origin === 'database' ? 'origin-database' : 'origin-web';
            const originText = image.origin === 'database' ? 'DB' : 'Web';
            
            const originBadge = document.createElement('div');
            originBadge.className = `image-origin-badge ${originClass}`;
            originBadge.textContent = originText;
            card.appendChild(originBadge);
        }
        
        // Add similarity badge if available
        if (image.similarity !== undefined) {
            const badge = document.createElement('div');
            badge.className = 'similarity-badge';
            badge.textContent = `${image.similarity}%`;
            card.appendChild(badge);
        }
        
        const img = document.createElement('img');
        img.src = image.source;
        img.alt = image.label;
        img.className = 'w-full h-40 object-cover rounded-lg mb-3 image-zoom';
        
        // Mark placeholder images visually
        if (image.is_placeholder) {
            img.classList.add('opacity-70');
        }
        
        const label = document.createElement('p');
        label.className = 'text-gray-800 font-medium truncate';
        label.textContent = image.label;
        
        card.appendChild(img);
        card.appendChild(label);
        
        // Add click event to analyze this image (unless it's a placeholder)
        if (!image.is_placeholder) {
            card.addEventListener('click', () => {
                analyzeImageUrl(image.source);
            });
        }
        
        similarContainer.appendChild(card);
    });
}

// Initialize paste handling for base64 images
function initializePasteHandling() {
    const urlInput = document.getElementById('url-input');
    
    // Add paste event listener if the URL input exists
    if (urlInput) {
        document.addEventListener('paste', function(e) {
            // Only process paste events when URL tab is active
            const urlTab = document.getElementById('url-tab');
            if (urlTab && urlTab.classList.contains('hidden')) {
                return;
            }
            
            // Get clipboard items
            const clipboardItems = e.clipboardData.items;
            
            for (let i = 0; i < clipboardItems.length; i++) {
                // Check if the clipboard item is an image
                if (clipboardItems[i].type.indexOf('image') !== -1) {
                    // Get the blob from clipboard
                    const blob = clipboardItems[i].getAsFile();
                    
                    // Create a FileReader to read the image as data URL
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        // Set the data URL to the URL input
                        urlInput.value = event.target.result;
                        
                        // Show a notification that an image was pasted
                        showNotification('Image pasted! Click "Analyze" to process it.', 'info');
                    };
                    reader.readAsDataURL(blob);
                    
                    // Prevent the default paste behavior
                    e.preventDefault();
                    break;
                }
            }
        });
    }
}

// Handle image file upload
function handleImageFile(file, numImages = 5, model = 'mistral:latest', safeSearch = true) {
    // Show loading state
    showLoading('Analyzing image...');
    
    // Create form data
    const formData = new FormData();
    formData.append('imageFile', file);
    formData.append('numImages', numImages);
    formData.append('model', model);
    formData.append('safeSearch', safeSearch);
    
    // Send file to server for analysis
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Failed to analyze image');
            });
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        if (data.success) {
            displayResults(data, model);
        } else {
            showError(data.error || 'Unknown error occurred while analyzing image');
        }
    })
    .catch(error => {
        console.error('Error analyzing image:', error);
        hideLoading();
        showError(error.message || 'Failed to analyze image. Please try again.');
    });
}

// Enhanced image URL analyzer
function analyzeImageUrl(url, numImages = 5, model = 'mistral:latest', safeSearch = true) {
    // Check if URL is empty
    if (!url || url.trim() === '') {
        showNotification('Please enter a valid image URL', 'error');
        return;
    }
    
    showLoading('Analyzing image URL...');
    
    // Create form data
    const formData = new FormData();
    formData.append('imageUrl', url);
    formData.append('numImages', numImages);
    formData.append('model', model);
    formData.append('safeSearch', safeSearch);
    
    // Send URL to server for analysis
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Failed to analyze image');
            });
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        if (data.success) {
            displayResults(data, model);
        } else {
            showError(data.error || 'Unknown error occurred while analyzing image');
        }
    })
    .catch(error => {
        console.error('Error analyzing image URL:', error);
        hideLoading();
        showError(error.message || 'Failed to analyze image URL. Please try again.');
    });
}

// Display analysis results
function displayResults(data, model = 'mistral:latest') {
    const resultsSection = document.getElementById('results');
    const searchResults = document.getElementById('search-results');
    const originalImage = document.getElementById('original-image');
    const detectedLabel = document.getElementById('detected-label');
    const confidenceScore = document.getElementById('confidence-score');
    const conceptType = document.getElementById('concept-type');
    const conceptDescription = document.getElementById('concept-description');
    const conceptFacts = document.getElementById('concept-facts');
    const relatedConcepts = document.getElementById('related-concepts');
    const knowledgeContent = document.getElementById('knowledge-content');
    const noKnowledge = document.getElementById('no-knowledge');
    const loadingInsights = document.getElementById('loading-insights');
    const modelUsed = document.getElementById('model-used');
    const numImagesResults = document.getElementById('num-images-results');
    
    // Hide search results and show analysis results
    if (searchResults) searchResults.classList.add('hidden');
    if (resultsSection) resultsSection.classList.remove('hidden');
    
    // Update original image section
    if (originalImage) originalImage.src = data.image.source;
    if (detectedLabel) detectedLabel.textContent = data.image.label;
    if (confidenceScore) confidenceScore.textContent = `${data.image.confidence}%`;
    
    // Update the number of similar images input
    if (numImagesResults) {
        numImagesResults.value = data.similar_images.length;
    }
    
    // Set model used
    if (modelUsed) {
        const modelName = model.split(':')[0].charAt(0).toUpperCase() + model.split(':')[0].slice(1);
        modelUsed.textContent = modelName;
    }
    
    // Update knowledge section
    if (conceptType && conceptDescription && conceptFacts && relatedConcepts && knowledgeContent && noKnowledge && loadingInsights) {
        // Hide loading and show content
        loadingInsights.classList.add('hidden');
        
        if (data.knowledge && Object.keys(data.knowledge).length > 0) {
            knowledgeContent.classList.remove('hidden');
            noKnowledge.classList.add('hidden');
            
            // Update type and description
            conceptType.textContent = data.knowledge.type || 'Unknown';
            conceptDescription.textContent = data.knowledge.description || 'No description available.';
            
            // Update facts
            conceptFacts.innerHTML = '';
            if (data.knowledge.facts && data.knowledge.facts.length > 0) {
                data.knowledge.facts.forEach(fact => {
                    const li = document.createElement('li');
                    li.textContent = fact;
                    conceptFacts.appendChild(li);
                });
            } else {
                const noFacts = document.createElement('li');
                noFacts.className = 'text-gray-500';
                noFacts.textContent = 'No facts available.';
                conceptFacts.appendChild(noFacts);
            }
            
            // Update related concepts
            relatedConcepts.innerHTML = '';
            
            if (data.knowledge.relations && data.knowledge.relations.length > 0) {
                data.knowledge.relations.forEach((relation, index) => {
                    // Add animation delay
                    const delay = index * 0.1;
                    
                    const badge = document.createElement('div');
                    badge.className = 'bg-indigo-100 text-indigo-800 rounded-full px-3 py-1 text-sm flex items-center knowledge-item';
                    badge.style.animationDelay = `${delay}s`;
                    
                    const icon = document.createElement('i');
                    icon.className = 'fas fa-link mr-2 text-indigo-600 text-xs';
                    
                    const text = document.createElement('span');
                    text.textContent = `${relation.relationship}: ${relation.related_concept}`;
                    
                    badge.appendChild(icon);
                    badge.appendChild(text);
                    relatedConcepts.appendChild(badge);
                });
            } else {
                const noConcepts = document.createElement('p');
                noConcepts.className = 'text-gray-500';
                noConcepts.textContent = 'No related concepts found.';
                relatedConcepts.appendChild(noConcepts);
            }
        } else {
            knowledgeContent.classList.add('hidden');
            noKnowledge.classList.remove('hidden');
        }
    }
    
    // Display similar images
    displaySimilarImages(data.similar_images);
    
    // Scroll to results
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize web scraping button
function initializeWebScrapingButton() {
    const scanPageBtn = document.getElementById('scan-page-btn');
    
    if (scanPageBtn) {
        scanPageBtn.addEventListener('click', () => {
            // Get the search input value
            const searchInput = document.getElementById('search-input');
            const searchTerm = searchInput ? searchInput.value.trim() : '';
            
            if (searchTerm) {
                showLoading('Searching web for images...');
                
                fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        term: searchTerm,
                        numWebResults: 10 // More web results for this specific search
                    })
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.success) {
                        displaySearchResults(data);
                    } else {
                        showError(data.error || 'Web search failed. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('Error in web search:', error);
                    hideLoading();
                    showError('Web search failed. Please try again.');
                });
            } else {
                showNotification('Please enter a search term first', 'error');
            }
        });
    }
}

// Show loading overlay
function showLoading(message = 'Loading...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        // Update loading message if provided
        const loadingMessage = loadingOverlay.querySelector('#loading-message');
        if (loadingMessage) {
            loadingMessage.textContent = message;
        }
        
        loadingOverlay.classList.remove('hidden');
    }
}

// Hide loading overlay
function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        loadingOverlay.classList.add('hidden');
    }
}

// Show error message
function showError(message) {
    // Create a toast-like error message
    const errorToast = document.createElement('div');
    errorToast.className = 'fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
    errorToast.textContent = message;
    
    document.body.appendChild(errorToast);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorToast.classList.add('opacity-0');
        errorToast.style.transition = 'opacity 0.5s ease';
        
        // Remove from DOM after fade out
        setTimeout(() => {
            if (document.body.contains(errorToast)) {
                document.body.removeChild(errorToast);
            }
        }, 500);
    }, 5000);
}

// Show a notification toast
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    
    // Set toast classes based on type
    let bgColor = 'bg-indigo-500';
    if (type === 'error') bgColor = 'bg-red-500';
    else if (type === 'success') bgColor = 'bg-green-500';
    else if (type === 'warning') bgColor = 'bg-yellow-500';
    
    toast.className = `fixed bottom-4 left-1/2 transform -translate-x-1/2 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg z-50`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Remove after 4 seconds
    setTimeout(() => {
        toast.classList.add('opacity-0');
        toast.style.transition = 'opacity 0.5s ease';
        
        // Remove from DOM after fade out
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 500);
    }, 4000);
}


// Initialize WebSocket connection
function initializeWebSocketConnection() {
    // Check if socket.io is loaded
    if (typeof io !== 'undefined') {
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to WebSocket server');
        });
        
        socket.on('log_message', function(data) {
            console.log(`[${data.level || 'info'}] ${data.message}`);
            
            // Update loading message if there's a loading overlay showing
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage && document.getElementById('loading-overlay').classList.contains('hidden') === false) {
                loadingMessage.textContent = data.message;
            }
        });
    } else {
        console.warn('Socket.io not loaded, WebSocket functionality will not be available');
    }
}

// Connect to WebSocket server
const socket = io();

// Initialize terminal
const terminalPanel = document.getElementById('terminal-panel');
const terminalHeader = document.getElementById('terminal-header');
const minimizeTerminal = document.getElementById('minimize-terminal');
const miniTerminalInput = document.getElementById('mini-terminal-input');
const miniTerminalOutput = document.getElementById('mini-terminal-output');

// Terminal toggle
let terminalOpen = false;

terminalHeader.addEventListener('click', function(e) {
    if (e.target.closest('#minimize-terminal')) return;
    
    if (terminalOpen) {
        terminalPanel.style.transform = 'translateY(100%)';
    } else {
        terminalPanel.style.transform = 'translateY(0)';
    }
    terminalOpen = !terminalOpen;
});

minimizeTerminal.addEventListener('click', function() {
    terminalPanel.style.transform = 'translateY(100%)';
    terminalOpen = false;
});

// Handle terminal input
miniTerminalInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        const command = miniTerminalInput.value.trim();
        if (command) {
            // Add command to output
            const commandElement = document.createElement('div');
            commandElement.className = 'flex';
            commandElement.innerHTML = `<span class="text-green-400 mr-2">$</span><span>${command}</span>`;
            miniTerminalOutput.appendChild(commandElement);
            
            // Send command to server
            socket.emit('execute_command', { command: command });
            
            // Clear input
            miniTerminalInput.value = '';
        }
    }
});

// Handle command results
socket.on('command_result', function(data) {
    const resultElement = document.createElement('div');
    resultElement.className = 'text-blue-300 whitespace-pre-wrap mb-2';
    resultElement.textContent = data.result;
    miniTerminalOutput.appendChild(resultElement);
    
    // Scroll to bottom
    const miniTerminal = document.getElementById('mini-terminal');
    miniTerminal.scrollTop = miniTerminal.scrollHeight;
});

// Handle log messages for notifications
socket.on('log_message', function(data) {
    // Show notifications for important logs
    if (data.level === 'error') {
        showError(data.message);
    } else if (data.level === 'warning') {
        showNotification(data.message, 'warning');
    } else if (data.message.includes('complete') || data.message.includes('success')) {
        showNotification(data.message, 'success');
    }
});