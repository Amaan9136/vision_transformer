document.addEventListener('DOMContentLoaded', function() {
    // Initialize socket connection
    const socket = io();
    
    // Navigation functionality
    initNavigation();
    
    // Tab switching
    initializeTabs();
    
    // Drag and drop
    initializeDragAndDrop();
    
    // Terminal functionality
    initializeTerminal(socket);
    
    // Initialize number inputs
    initializeNumberInputs();
    
    // Initialize settings toggles
    initializeSettingsToggles();
    
    // Initialize image upload/URL
    initializeImageHandling();
    
    // Initialize search
    initializeSearch();
    
    // Initialize visualization and explanation sections
    initializeVisualizationSection();
    initializeExplanationSection();
    
    // Initialize log container
    initializeLogContainer(socket);
    
    // Initialize dialogs
    initializeDialogs();
});

// Initialize navigation between sections
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.page-section');
    
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            navButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Get section ID
            const sectionId = button.id.replace('nav-', 'section-');
            
            // Hide all sections
            sections.forEach(section => section.classList.add('hidden'));
            
            // Show selected section
            document.getElementById(sectionId).classList.remove('hidden');
        });
    });
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

// Initialize terminal functionality
function initializeTerminal(socket) {
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
        if (miniTerminal) {
            miniTerminal.scrollTop = miniTerminal.scrollHeight;
        }
    });
    
    // Same for the main terminal on logs page
    const terminalInput = document.getElementById('terminal-input');
    const terminalOutput = document.getElementById('terminal-output');
    
    if (terminalInput && terminalOutput) {
        terminalInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const command = terminalInput.value.trim();
                if (command) {
                    // Add command to output
                    const commandElement = document.createElement('div');
                    commandElement.className = 'flex mb-1';
                    commandElement.innerHTML = `<span class="prompt mr-2">$</span><span class="command">${command}</span>`;
                    terminalOutput.appendChild(commandElement);
                    
                    // Send command to server
                    socket.emit('execute_command', { command: command });
                    
                    // Clear input
                    terminalInput.value = '';
                }
            }
        });
    }
}

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

// Initialize image handling
function initializeImageHandling() {
    // URL button handling
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
                urlBtn.click();
            }
        });
    }
    
    // Initialize web scraping button
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
    
    // Initialize Find More Similar Images button
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
    
    // Button to explain transformer process
    const explainProcessBtn = document.getElementById('btn-explain-process');
    if (explainProcessBtn) {
        explainProcessBtn.addEventListener('click', () => {
            // Show transformer insights section
            const transformerInsights = document.getElementById('transformer-insights');
            if (transformerInsights) {
                transformerInsights.classList.remove('hidden');
                
                // Get the image label
                const detectedLabel = document.getElementById('detected-label');
                if (detectedLabel && detectedLabel.textContent) {
                    // Get transformer explanation for this type
                    fetch(`/transformer_explanation?type=${detectedLabel.textContent}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                // Update transformer insight text
                                const insightText = document.getElementById('transformer-insight-text');
                                if (insightText) {
                                    insightText.textContent = data.explanation.custom_insight || 
                                                              "Vision Transformers divide the image into fixed-size patches and process them using self-attention to understand global relationships.";
                                }
                                
                                // Show transformation pipeline
                                const transformerPipelineImg = document.getElementById('transformer-pipeline-img');
                                const originalImage = document.getElementById('original-image');
                                if (transformerPipelineImg && originalImage) {
                                    const imageData = originalImage.getAttribute('data-transformation-pipeline');
                                    if (imageData) {
                                        transformerPipelineImg.src = imageData;
                                    }
                                }
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching transformer explanation:', error);
                        });
                }
                
                // Scroll to the section
                transformerInsights.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    // Button to view attention maps
    const viewAttentionBtn = document.getElementById('btn-view-attention');
    if (viewAttentionBtn) {
        viewAttentionBtn.addEventListener('click', () => {
            const attentionDialog = document.getElementById('attention-maps-dialog');
            const originalImage = document.getElementById('original-image');
            
            if (attentionDialog && originalImage) {
                // Get attention map path from data attribute
                const attentionMapPath = originalImage.getAttribute('data-attention-maps');
                if (attentionMapPath) {
                    // Set the image source
                    const combinedAttentionMap = document.getElementById('combined-attention-map');
                    if (combinedAttentionMap) {
                        combinedAttentionMap.src = attentionMapPath;
                    }
                    
                    // Get the image label
                    const detectedLabel = document.getElementById('detected-label');
                    if (detectedLabel && detectedLabel.textContent) {
                        // Get attention explanation for this type
                        fetch(`/attention_explanation?type=${detectedLabel.textContent}`)
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    // Update attention explanation text
                                    const attentionExplanation = document.getElementById('attention-explanation');
                                    if (attentionExplanation) {
                                        attentionExplanation.textContent = data.explanation.specific_explanation || 
                                                                          "The attention mechanism allows the model to focus on different parts of the image based on their relevance to the task.";
                                    }
                                }
                            })
                            .catch(error => {
                                console.error('Error fetching attention explanation:', error);
                            });
                    }
                    
                    // Show the dialog
                    attentionDialog.classList.remove('hidden');
                } else {
                    showNotification('Attention maps not available for this image', 'error');
                }
            }
        });
    }
    
    // Button to view feature maps
    const viewFeaturesBtn = document.getElementById('btn-view-features');
    if (viewFeaturesBtn) {
        viewFeaturesBtn.addEventListener('click', () => {
            const featureMapsDialog = document.getElementById('feature-maps-dialog');
            const originalImage = document.getElementById('original-image');
            
            if (featureMapsDialog && originalImage) {
                // Get feature maps path from data attribute
                const featureMapsPath = originalImage.getAttribute('data-feature-maps');
                if (featureMapsPath) {
                    // Set the image source
                    const featureMapsImg = document.getElementById('feature-maps-img');
                    if (featureMapsImg) {
                        featureMapsImg.src = featureMapsPath;
                    }
                    
                    // Get the image label
                    const detectedLabel = document.getElementById('detected-label');
                    if (detectedLabel && detectedLabel.textContent) {
                        // Get feature maps explanation for this type
                        fetch(`/feature_maps_explanation?type=${detectedLabel.textContent}`)
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    // Update feature explanation text
                                    const featureExplanation = document.getElementById('feature-explanation');
                                    if (featureExplanation) {
                                        featureExplanation.textContent = data.explanation.specific_explanation || 
                                                                        "Feature maps represent the learned features at different layers of the model, capturing both low-level and high-level patterns.";
                                    }
                                }
                            })
                            .catch(error => {
                                console.error('Error fetching feature maps explanation:', error);
                            });
                    }
                    
                    // Show the dialog
                    featureMapsDialog.classList.remove('hidden');
                } else {
                    showNotification('Feature maps not available for this image', 'error');
                }
            }
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

// Initialize visualization section
function initializeVisualizationSection() {
    const visualizationType = document.getElementById('visualization-type');
    
    if (visualizationType) {
        visualizationType.addEventListener('change', () => {
            const selectedType = visualizationType.value;
            
            // Update visualization based on selected type
            updateVisualization(selectedType);
        });
        
        // Initialize with default visualization
        updateVisualization('attention');
    }
}

// Update visualization based on type
function updateVisualization(type) {
    const visualizationImage = document.getElementById('visualization-image');
    const visualizationExplanation = document.getElementById('visualization-explanation');
    
    if (!visualizationImage || !visualizationExplanation) return;
    
    // Set default image path
    let imagePath = '/static/images/vit_architecture.png';
    let explanationTitle = 'How Attention Works in Vision Transformers';
    let explanationText = 'Vision Transformers process images by dividing them into fixed-size patches and treating them as a sequence. The self-attention mechanism allows the model to focus on the most relevant parts of the image for the task at hand.';
    
    // Update content based on type
    switch (type) {
        case 'attention':
            imagePath = '/static/images/attention_visualization.png';
            explanationTitle = 'How Attention Works in Vision Transformers';
            explanationText = 'Vision Transformers use multi-head self-attention to determine which parts of an image are most relevant. Each attention head can focus on different aspects of the image, allowing the model to capture complex relationships.';
            break;
        case 'features':
            imagePath = '/static/images/feature_maps.png';
            explanationTitle = 'Feature Maps in Vision Transformers';
            explanationText = 'Unlike CNNs with explicit feature maps, Vision Transformers produce features through self-attention and MLP operations. Each transformer layer produces hidden state vectors that act as feature representations.';
            break;
        case 'pipeline':
            imagePath = '/static/images/vit_pipeline.png';
            explanationTitle = 'Transformation Pipeline';
            explanationText = 'The Vision Transformer pipeline involves dividing the image into patches, embedding these patches, adding positional embeddings, and processing through transformer layers to produce the final classification.';
            break;
        case 'comparison':
            imagePath = '/static/images/cnn_vs_vit.png';
            explanationTitle = 'Vision Transformers vs. CNNs';
            explanationText = 'Vision Transformers differ from CNNs in how they process images. While CNNs use convolutional filters with local receptive fields, ViTs treat images as sequences of patches with global attention mechanism.';
            break;
    }
    
    // Update image
    visualizationImage.innerHTML = `<img src="${imagePath}" alt="Visualization" class="max-w-full rounded-lg shadow-lg">`;
    
    // Update explanation
    visualizationExplanation.innerHTML = `
        <h3 class="text-xl font-semibold text-gray-800 mb-4">${explanationTitle}</h3>
        <p class="text-gray-600">${explanationText}</p>
    `;
    
    // Fetch additional information from the server
    fetch(`/${type === 'comparison' ? 'transformer_comparison' : type + '_explanation'}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update explanation with additional information
                visualizationExplanation.innerHTML += `
                    <p class="text-gray-600 mt-4">${data.explanation || data.comparison || data.insight || ''}</p>
                `;
            }
        })
        .catch(error => {
            console.error(`Error fetching ${type} information:`, error);
        });
}

// Initialize explanation section
function initializeExplanationSection() {
    const explanationType = document.getElementById('explanation-type');
    
    if (explanationType) {
        explanationType.addEventListener('change', () => {
            const selectedType = explanationType.value;
            
            // Update explanation based on selected type
            updateExplanation(selectedType);
        });
        
        // Initialize with default explanation
        updateExplanation('general');
    }
}

// Update explanation based on type
function updateExplanation(type) {
    const explanationContent = document.getElementById('explanation-content');
    
    if (!explanationContent) return;
    
    // Show loading indicator
    explanationContent.innerHTML = '<div class="flex justify-center py-8"><div class="loader-spinner mr-3"></div><p>Loading explanation...</p></div>';
    
    // Map type to API endpoint
    let endpoint = 'transformer_explanation';
    if (type === 'attention') endpoint = 'attention_explanation';
    else if (type === 'comparison') endpoint = 'transformer_comparison';
    else if (type === 'tasks') endpoint = 'task_specific_insights?task=classification';
    else if (type === 'deployment') endpoint = 'deployment_considerations';
    
    // Fetch explanation data
    fetch(`/${endpoint}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Create content based on type
                let content = '';
                
                if (type === 'general') {
                    content = `
                        <h3 class="text-xl font-semibold text-gray-800 mb-4">${data.explanation.title || 'How Vision Transformers Work'}</h3>
                        <p class="text-gray-600 mb-4">${data.explanation.introduction || ''}</p>
                        <h4 class="text-lg font-medium text-gray-700 mb-3">Key Stages:</h4>
                        <ul class="list-disc pl-5 mb-4">
                            ${data.explanation.stages ? data.explanation.stages.map(stage => `
                                <li class="mb-2">
                                    <span class="font-medium">${stage.name}:</span> ${stage.description}
                                </li>
                            `).join('') : ''}
                        </ul>
                        <h4 class="text-lg font-medium text-gray-700 mb-3">Key Differences from CNNs:</h4>
                        <ul class="list-disc pl-5">
                            ${data.explanation.key_differences ? data.explanation.key_differences.map(diff => `
                                <li class="mb-2">${diff}</li>
                            `).join('') : ''}
                        </ul>
                    `;
                } else if (type === 'comparison') {
                    content = `
                        <h3 class="text-xl font-semibold text-gray-800 mb-4">${data.comparison.title || 'Vision Transformers vs. CNNs'}</h3>
                        <p class="text-gray-600 mb-4">${data.comparison.introduction || ''}</p>
                        <div class="overflow-x-auto">
                            <table class="min-w-full bg-white border border-gray-200 rounded-lg">
                                <thead>
                                    <tr>
                                        <th class="px-6 py-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Aspect</th>
                                        <th class="px-6 py-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CNN</th>
                                        <th class="px-6 py-3 border-b border-gray-200 bg-gray-100 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Vision Transformer</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.comparison.comparison_points ? data.comparison.comparison_points.map(point => `
                                        <tr>
                                            <td class="px-6 py-4 whitespace-nowrap border-b border-gray-200 font-medium">${point.aspect}</td>
                                            <td class="px-6 py-4 border-b border-gray-200">${point.cnn}</td>
                                            <td class="px-6 py-4 border-b border-gray-200">${point.vit}</td>
                                        </tr>
                                    `).join('') : ''}
                                </tbody>
                            </table>
                        </div>
                        <p class="text-gray-600 mt-4">${data.comparison.conclusion || ''}</p>
                    `;
                } else if (type === 'tasks') {
                    content = `
                        <h3 class="text-xl font-semibold text-gray-800 mb-4">Vision Transformers for Different Tasks</h3>
                        <p class="text-gray-600 mb-4">${data.insights.general_insight || ''}</p>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                            <div class="bg-white p-4 rounded-lg shadow">
                                <h4 class="text-lg font-medium text-gray-700 mb-3">Classification</h4>
                                <p class="text-gray-600">${data.insights.specific_insight || ''}</p>
                            </div>
                            
                            <div class="bg-white p-4 rounded-lg shadow">
                                <h4 class="text-lg font-medium text-gray-700 mb-3">Object Detection</h4>
                                <p class="text-gray-600">Vision Transformers have been adapted for object detection using models like DETR (DEtection TRansformer), which reformulates detection as a direct set prediction problem.</p>
                            </div>
                            
                            <div class="bg-white p-4 rounded-lg shadow">
                                <h4 class="text-lg font-medium text-gray-700 mb-3">Segmentation</h4>
                                <p class="text-gray-600">For segmentation tasks, Vision Transformers provide pixel-level predictions by establishing relationships between all image regions, helping resolve ambiguities at object boundaries.</p>
                            </div>
                            
                            <div class="bg-white p-4 rounded-lg shadow">
                                <h4 class="text-lg font-medium text-gray-700 mb-3">Image Generation</h4>
                                <p class="text-gray-600">Vision Transformers can model complex dependencies between image elements for generation tasks, capturing global structure and coherence across the generated image.</p>
                            </div>
                        </div>
                    `;
                } else if (type === 'deployment') {
                    content = `
                        <h3 class="text-xl font-semibold text-gray-800 mb-4">${data.considerations.title || 'Deployment Considerations'}</h3>
                        <p class="text-gray-600 mb-4">${data.considerations.introduction || ''}</p>
                        
                        <div class="space-y-4 mt-4">
                            ${data.considerations.considerations ? data.considerations.considerations.map(consideration => `
                                <div class="bg-white p-4 rounded-lg shadow">
                                    <h4 class="text-lg font-medium text-gray-700 mb-2">${consideration.aspect}</h4>
                                    <p class="text-gray-600">${consideration.description}</p>
                                </div>
                            `).join('') : ''}
                        </div>
                        
                        <div class="bg-indigo-50 p-4 rounded-lg mt-6">
                            <h4 class="text-lg font-medium text-indigo-700 mb-2">Conclusion</h4>
                            <p class="text-indigo-600">${data.considerations.conclusion || ''}</p>
                        </div>
                    `;
                } else if (type === 'attention') {
                    content = `
                        <h3 class="text-xl font-semibold text-gray-800 mb-4">Attention Mechanism in Vision Transformers</h3>
                        <p class="text-gray-600 mb-4">${data.explanation.general_explanation || ''}</p>
                        
                        <div class="bg-gray-50 p-4 rounded-lg my-4">
                            <h4 class="text-lg font-medium text-gray-700 mb-3">Attention Heads</h4>
                            <p class="text-gray-600">${data.explanation.attention_heads || 'Different attention heads in the model focus on different aspects of the image.'}</p>
                        </div>
                        
                        <div class="bg-white p-4 rounded-lg shadow">
                            <h4 class="text-lg font-medium text-gray-700 mb-3">Image-Specific Attention</h4>
                            <p class="text-gray-600">${data.explanation.specific_explanation || ''}</p>
                        </div>
                    `;
                }
                
                // Update explanation content
                explanationContent.innerHTML = content;
            } else {
                explanationContent.innerHTML = '<p class="text-red-500">Failed to load explanation. Please try again.</p>';
            }
        })
        .catch(error => {
            console.error(`Error fetching ${type} explanation:`, error);
            explanationContent.innerHTML = '<p class="text-red-500">Error loading explanation. Please try again.</p>';
        });
}

// Initialize log container
function initializeLogContainer(socket) {
    const logContainer = document.getElementById('log-container');
    
    if (logContainer) {
        // Clear initial message on connection
        socket.on('connect', function() {
            logContainer.innerHTML = '';
            addLogMessage('Connected to server', 'info');
        });
        
        // Handle log messages
        socket.on('log_message', function(data) {
            addLogMessage(data.message, data.level || 'info');
        });
        
        // Function to add log message
        function addLogMessage(message, level) {
            const logEntry = document.createElement('div');
            logEntry.className = `mb-1 log-${level}`;
            
            const timestamp = new Date().toLocaleTimeString();
            logEntry.innerHTML = `[${timestamp}] ${message}`;
            
            logContainer.appendChild(logEntry);
            
            // Scroll to bottom
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }
}

// Initialize dialogs
function initializeDialogs() {
    // Attention maps dialog
    const attentionDialog = document.getElementById('attention-maps-dialog');
    const closeAttentionDialog = document.getElementById('close-attention-dialog');
    
    if (attentionDialog && closeAttentionDialog) {
        closeAttentionDialog.addEventListener('click', () => {
            attentionDialog.classList.add('hidden');
        });
        
        // Close when clicking outside the content
        attentionDialog.addEventListener('click', (e) => {
            if (e.target === attentionDialog) {
                attentionDialog.classList.add('hidden');
            }
        });
    }
    
    // Feature maps dialog
    const featureMapsDialog = document.getElementById('feature-maps-dialog');
    const closeFeatureDialog = document.getElementById('close-feature-dialog');
    
    if (featureMapsDialog && closeFeatureDialog) {
        closeFeatureDialog.addEventListener('click', () => {
            featureMapsDialog.classList.add('hidden');
        });
        
        // Close when clicking outside the content
        featureMapsDialog.addEventListener('click', (e) => {
            if (e.target === featureMapsDialog) {
                featureMapsDialog.classList.add('hidden');
            }
        });
    }
}

// Function to perform search
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

// Function to display search results
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

// Function to hide search results
function hideSearchResults() {
    const searchResults = document.getElementById('search-results');
    const resultsSection = document.getElementById('results');
    
    if (searchResults) searchResults.classList.add('hidden');
    if (resultsSection) resultsSection.classList.remove('hidden');
}

// Function to handle image file upload
function handleImageFile(file, numImages = 5, model = 'mistral:latest', safeSearch = true) {
    // Show loading state
    showLoading('Analyzing image...');
    
    // Create form data
    const formData = new FormData();
    formData.append('imageFile', file);
    formData.append('numImages', numImages);
    formData.append('model', model);
    formData.append('safeSearch', safeSearch);
    formData.append('generateVisualizations', true);  // Always generate visualizations
    
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

// Function to analyze image URL
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
    formData.append('generateVisualizations', true);  // Always generate visualizations
    
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

// Function to display analysis results
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
    if (originalImage) {
        originalImage.src = data.image.source;
        
        // Store visualization paths as data attributes
        if (data.image.attention_maps) {
            originalImage.setAttribute('data-attention-maps', data.image.attention_maps);
        }
        if (data.image.feature_maps) {
            originalImage.setAttribute('data-feature-maps', data.image.feature_maps);
        }
        if (data.image.transformation_pipeline) {
            originalImage.setAttribute('data-transformation-pipeline', data.image.transformation_pipeline);
        }
    }
    
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
    
    // Hide transformer insights section initially
    const transformerInsights = document.getElementById('transformer-insights');
    if (transformerInsights) {
        transformerInsights.classList.add('hidden');
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
    
    // Update transformer insight text if available
    const transformerInsightText = document.getElementById('transformer-insight-text');
    if (transformerInsightText && data.knowledge && data.knowledge.transformer_insight) {
        transformerInsightText.textContent = data.knowledge.transformer_insight;
    }
    
    // Display similar images
    displaySimilarImages(data.similar_images);
    
    // Scroll to results
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// Function to display similar images
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
    
    // Show no similar message if no images
    if (images.length === 0 && noSimilar) {
        noSimilar.classList.remove('hidden');
    }
}

// Function to fetch more similar images
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

// Function to show loading overlay
function showLoading(message = 'Loading...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        // Update loading message if provided
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.textContent = message;
        }
        
        loadingOverlay.classList.remove('hidden');
    }
}

// Function to hide loading overlay
function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        loadingOverlay.classList.add('hidden');
    }
}

// Function to show error message
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

// Function to show a notification toast
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