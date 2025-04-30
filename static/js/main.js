// Main JavaScript for Vision Knowledge Explorer

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
    
});

// Initialize tab switching functionality
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
                handleImageFile(files[0]);
            }
        });
        
        // Handle browse files click
        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageFile(e.target.files[0]);
            }
        });
        
        // Handle image file upload
        function handleImageFile(file) {
            // Show loading state
            showLoading('Analyzing image...');
            
            // Create form data
            const formData = new FormData();
            formData.append('imageFile', file);
            
            // Send file to server for analysis
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResults(data);
            })
            .catch(error => {
                console.error('Error analyzing image:', error);
                hideLoading();
                showError('Failed to analyze image. Please try again.');
            });
        }
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
        body: JSON.stringify({ term: searchTerm })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        displaySearchResults(data);
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
        
        if (data.success && data.results && data.results.length > 0) {
            // Hide no results message
            noResults.classList.add('hidden');
            
            // Add each result to the container with animation delay
            data.results.forEach((result, index) => {
                const delay = index * 0.1;
                
                const card = document.createElement('div');
                card.className = 'image-card glass-panel p-4 rounded-lg overflow-hidden relative';
                card.style.animationDelay = `${delay}s`;
                
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
    // Create a "Scan Page" button
    const scanPageBtn = document.createElement('button');
    scanPageBtn.id = 'scan-page-btn';
    scanPageBtn.className = 'btn-primary px-6 py-2 rounded-full mb-4';
    scanPageBtn.innerHTML = '<i class="fas fa-spider mr-2"></i> Scan Page for Images';
    
    // Add the button to URL tab, just before the existing content
    const urlTab = document.getElementById('url-tab');
    if (urlTab) {
        urlTab.insertBefore(scanPageBtn, urlTab.firstChild);
        
        // Add click event handler
        scanPageBtn.addEventListener('click', () => {
            scrapeImagesFromPage();
        });
    }
    
    // Add URL input functionality
    const urlBtn = document.getElementById('url-btn');
    const urlInput = document.getElementById('url-input');
    
    if (urlBtn && urlInput) {
        urlBtn.addEventListener('click', () => {
            const imageUrl = urlInput.value.trim();
            if (imageUrl) {
                analyzeImageUrl(imageUrl);
            }
        });
        
        // Handle enter key in URL input
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const imageUrl = urlInput.value.trim();
                if (imageUrl) {
                    analyzeImageUrl(imageUrl);
                }
            }
        });
    }
}

// Scrape images from the current page
function scrapeImagesFromPage() {
    showLoading('Scanning page for images...');
    
    // In a real implementation, this would scan the DOM for images
    // For our demo, we'll simulate with a delay and placeholder results
    setTimeout(() => {
        // Create a placeholder for image selections
        const scrapedImagesContainer = document.createElement('div');
        scrapedImagesContainer.id = 'scraped-images';
        scrapedImagesContainer.className = 'mt-6 border-t pt-6';
        
        const heading = document.createElement('h3');
        heading.textContent = 'Images found on this page';
        heading.className = 'text-lg font-semibold mb-4';
        
        const imagesGrid = document.createElement('div');
        imagesGrid.className = 'grid grid-cols-3 gap-4';
        
        // Add placeholder images (in a real app, these would be actual scraped images)
        const imageSources = [
            '/static/uploads/scraped_image_1.jpg',
            '/static/uploads/scraped_image_2.jpg',
            '/static/uploads/scraped_image_3.jpg',
            '/static/uploads/scraped_image_4.jpg',
            '/static/uploads/scraped_image_5.jpg',
            '/static/uploads/scraped_image_6.jpg'
        ];
        
        imageSources.forEach(src => {
            const imageWrapper = document.createElement('div');
            imageWrapper.className = 'relative border border-gray-200 rounded-lg overflow-hidden cursor-pointer';
            
            const img = document.createElement('img');
            img.src = src;
            img.className = 'w-full h-32 object-cover image-zoom';
            
            // Add click event to analyze this image
            imageWrapper.addEventListener('click', () => {
                analyzeImageUrl(src);
            });
            
            imageWrapper.appendChild(img);
            imagesGrid.appendChild(imageWrapper);
        });
        
        scrapedImagesContainer.appendChild(heading);
        scrapedImagesContainer.appendChild(imagesGrid);
        
        // Add to the URL tab
        const urlTab = document.getElementById('url-tab');
        
        // Remove any existing scraped images container
        const existingContainer = document.getElementById('scraped-images');
        if (existingContainer) {
            existingContainer.remove();
        }
        
        urlTab.appendChild(scrapedImagesContainer);
        
        hideLoading();
    }, 2000);
}

// Analyze an image URL
function analyzeImageUrl(url) {
    showLoading('Analyzing image...');
    
    // Create form data
    const formData = new FormData();
    formData.append('imageUrl', url);
    
    // Send URL to server for analysis
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        displayResults(data);
    })
    .catch(error => {
        console.error('Error analyzing image URL:', error);
        hideLoading();
        showError('Failed to analyze image URL. Please try again.');
    });
}

// Display analysis results
function displayResults(data) {
    if (!data.success) {
        showError(data.error || 'Failed to analyze image.');
        return;
    }
    
    // Get DOM elements
    const resultsSection = document.getElementById('results');
    const searchResults = document.getElementById('search-results');
    const originalImage = document.getElementById('original-image');
    const detectedLabel = document.getElementById('detected-label');
    const confidenceScore = document.getElementById('confidence-score');
    const conceptType = document.getElementById('concept-type');
    const conceptDescription = document.getElementById('concept-description');
    const relatedConcepts = document.getElementById('related-concepts');
    const knowledgeContent = document.getElementById('knowledge-content');
    const noKnowledge = document.getElementById('no-knowledge');
    const similarContainer = document.getElementById('similar-container');
    const noSimilar = document.getElementById('no-similar');
    
    // Hide search results and show analysis results
    if (searchResults) searchResults.classList.add('hidden');
    if (resultsSection) resultsSection.classList.remove('hidden');
    
    // Update original image section
    if (originalImage) originalImage.src = data.image.source;
    if (detectedLabel) detectedLabel.textContent = data.image.label;
    if (confidenceScore) confidenceScore.textContent = `${data.image.confidence}%`;
    
    // Update knowledge section
    if (conceptType && conceptDescription && relatedConcepts && knowledgeContent && noKnowledge) {
        if (data.knowledge && Object.keys(data.knowledge).length > 0) {
            knowledgeContent.classList.remove('hidden');
            noKnowledge.classList.add('hidden');
            
            conceptType.textContent = data.knowledge.type || 'Unknown';
            conceptDescription.textContent = data.knowledge.description || 'No description available.';
            
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
    
    // Update similar images section
    if (similarContainer && noSimilar) {
        similarContainer.innerHTML = '';
        
        if (data.similar_images && data.similar_images.length > 0) {
            noSimilar.classList.add('hidden');
            
            data.similar_images.forEach((image, index) => {
                // Add animation delay
                const delay = index * 0.1;
                
                const card = document.createElement('div');
                card.className = 'image-card glass-panel p-4 rounded-lg overflow-hidden relative';
                card.style.animationDelay = `${delay}s`;
                
                const img = document.createElement('img');
                img.src = image.source;
                img.alt = image.label;
                img.className = 'w-full h-40 object-cover rounded-lg mb-3 image-zoom';
                
                const badge = document.createElement('div');
                badge.className = 'similarity-badge';
                badge.textContent = `${image.similarity}%`;
                
                const label = document.createElement('p');
                label.className = 'text-gray-800 font-medium truncate';
                label.textContent = image.label;
                
                card.appendChild(img);
                card.appendChild(badge);
                card.appendChild(label);
                
                // Add click event to analyze this image
                card.addEventListener('click', () => {
                    analyzeImageUrl(image.source);
                });
                
                similarContainer.appendChild(card);
            });
        } else {
            noSimilar.classList.remove('hidden');
        }
    }
    
    // Scroll to results
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Show loading overlay
function showLoading(message = 'Loading...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        // Update loading message if provided
        const loadingMessage = loadingOverlay.querySelector('p');
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
            document.body.removeChild(errorToast);
        }, 500);
    }, 5000);
}

// Function to handle pasted content (for base64 images)
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

// Enhanced image URL analyzer with better error handling
function analyzeImageUrl(url) {
    // Check if URL is empty
    if (!url || url.trim() === '') {
        showNotification('Please enter a valid image URL', 'error');
        return;
    }
    
    showLoading('Analyzing image...');
    
    // Create form data
    const formData = new FormData();
    formData.append('imageUrl', url);
    
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
            displayResults(data);
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

// Show a notification toast
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    
    // Set toast classes based on type
    let bgColor = 'bg-indigo-500';
    if (type === 'error') bgColor = 'bg-red-500';
    else if (type === 'success') bgColor = 'bg-green-500';
    
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

// Update the DOMContentLoaded event in main.js to include the new function
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
    
    // Initialize paste handling for base64 images
    initializePasteHandling();
});