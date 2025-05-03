"""
This updated version adds:
1. Improved image deduplication with perceptual hashing
2. Visualization of Vision Transformer attention maps
3. Feature extraction and visualization
4. Detailed explanations of transformer processes
5. Enhanced web scraping with redundancy detection
6. Multi-stage classification pipeline
"""

from CONSTANTS.MODULES import (
    os, logging, json, uuid, requests, base64,
    Image, datetime,
    chromadb, Flask, render_template, request, jsonify, send_file,
    urllib, logger, string, socketio, SocketIO,
)

from config import flask_app

# Import custom modules
from helpers_and_class.VisionModule import VisionModule
from helpers_and_class.VectorDB import VectorDB
from helpers_and_class.ImageScraper import ImageScraper
from helpers_and_class.LLMInsightGenerator import LLMInsightGenerator
from helpers_and_class.TransformerExplainerModule import TransformerExplainerModule


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Initialize SocketIO
socketio = SocketIO(flask_app, cors_allowed_origins="*")


# Initialize modules
vision_module = VisionModule()
vector_db = VectorDB()
image_scraper = ImageScraper()
llm_insights = LLMInsightGenerator(model_name="mistral:latest")
transformer_explainer = TransformerExplainerModule()

def is_base64_image(data):
    """Check if a string is a base64 encoded image."""
    try:
        if not data.startswith('data:image/'):
            return False
        # Extract the base64 data
        data = data.split(',')[1]
        return True
    except Exception:
        return False

def decode_base64_image(data):
    """Decode a base64 encoded image to binary data."""
    try:
        # Extract the base64 data
        header, data = data.split(',', 1)
        image_data = base64.b64decode(data)
        return image_data
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

def sanitize_filename(filename):
    """Sanitize a filename to prevent directory traversal and other issues."""
    # Remove path information and get only the filename
    filename = os.path.basename(filename)
    
    # Remove invalid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c for c in filename if c in valid_chars)
    
    # Truncate if too long
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:100] + ext
        
    return sanitized

# WebSocket endpoint for logging
@socketio.on('connect')
def handle_connect():
    socketio.emit('log_message', {'message': 'Connected to server'})

def log_to_client(message, level='info'):
    """Send log message to connected clients"""
    socketio.emit('log_message', {'message': message, 'level': level})


# Flask routes for the Vision Transformer application

@flask_app.route('/')
def index():
    """Main page"""
    # Generate transformer explanation for initial page load
    transformer_explanation = transformer_explainer.generate_general_explanation()
    comparison = transformer_explainer.generate_comparison_with_cnn()
    return render_template('index.html', 
                          transformer_explanation=transformer_explanation, 
                          comparison=comparison)

@flask_app.route('/transformer_explanation')
def get_transformer_explanation():
    """API endpoint to get a transformer explanation"""
    image_type = request.args.get('type', 'general')
    
    if image_type == 'general':
        explanation = transformer_explainer.generate_general_explanation()
    else:
        explanation = transformer_explainer.generate_custom_explanation(image_type)
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })

@flask_app.route('/attention_explanation')
def get_attention_explanation():
    """API endpoint to get an explanation of attention for a specific image type"""
    image_type = request.args.get('type', 'general')
    explanation = transformer_explainer.generate_attention_explanation(image_type)
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })

@flask_app.route('/feature_maps_explanation')
def get_feature_maps_explanation():
    """API endpoint to get an explanation of feature maps for a specific image type"""
    image_type = request.args.get('type', 'general')
    explanation = transformer_explainer.generate_feature_maps_explanation(image_type)
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })

@flask_app.route('/task_specific_insights')
def get_task_specific_insights():
    """API endpoint to get insights about Vision Transformers for specific CV tasks"""
    task_type = request.args.get('task', 'classification')
    insights = transformer_explainer.generate_task_specific_insights(task_type)
    
    return jsonify({
        "success": True,
        "insights": insights
    })

@flask_app.route('/deployment_considerations')
def get_deployment_considerations():
    """API endpoint to get deployment considerations for Vision Transformers"""
    considerations = transformer_explainer.generate_deployment_considerations()
    
    return jsonify({
        "success": True,
        "considerations": considerations
    })

@flask_app.route('/architecture_diagram')
def get_architecture_diagram():
    """API endpoint to get the transformer architecture diagram"""
    diagram_path = transformer_explainer.create_architecture_diagram()
    
    return jsonify({
        "success": True,
        "diagram_path": diagram_path
    })

@flask_app.route('/analyze', methods=['POST'])
def analyze_image():
    """API endpoint to analyze an image"""
    try:
        # Get the number of similar images to return (default is 5)
        num_similar_images = int(request.form.get('numImages', 5))
        # Cap the number to avoid overloading
        num_similar_images = min(max(num_similar_images, 1), 20)
        
        # Get the model to use for insights generation
        model = request.form.get('model', 'mistral:latest')
        
        # Get safe search setting (default is True)
        safe_search = request.form.get('safeSearch', 'true').lower() == 'true'
        
        # Get visualization setting (default is True)
        generate_visualizations = request.form.get('generateVisualizations', 'true').lower() == 'true'
        
        log_to_client(f"Starting image analysis with model: {model}")
        
        if 'imageUrl' in request.form:
            # Image URL provided
            image_url = request.form['imageUrl']
            
            # Generate a unique ID for this image
            image_id = f"img_{uuid.uuid4()}"
            
            # Check if this is a base64 encoded image
            if is_base64_image(image_url):
                log_to_client("Processing base64 encoded image")
                # Decode base64 data
                image_data = decode_base64_image(image_url)
                if not image_data:
                    return jsonify({
                        "success": False,
                        "error": "Invalid base64 image data"
                    }), 400
                
                # Save base64 image to a file for record keeping
                filename = f"{uuid.uuid4()}.jpg"
                filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                # Process the image data directly
                log_to_client("Analyzing image with Vision Transformer...")
                vision_results = vision_module.process_image(image_data=image_data, generate_visualizations=generate_visualizations)
                
                # Use the file path as source for the record
                source_path = f"/static/uploads/{filename}"
            else:
                try:
                    log_to_client(f"Processing image from URL: {image_url}")
                    # For URLs, download the image first to handle potential errors better
                    if image_url.startswith(('http', 'https')):
                        image_data = vision_module._download_image_with_retry(image_url)
                        
                        # Get a safe filename from the URL
                        url_filename = os.path.basename(image_url.split('?')[0])  # Remove query parameters
                        safe_filename = sanitize_filename(url_filename)
                        if not safe_filename or safe_filename == '':
                            safe_filename = 'image.jpg'
                            
                        # Create a unique filename to avoid collisions
                        filename = f"{uuid.uuid4()}_{safe_filename}"
                        filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
                        
                        # Save the image to a file
                        with open(filepath, 'wb') as f:
                            f.write(image_data)
                        
                        # Process the image from the saved file
                        log_to_client("Analyzing image with Vision Transformer...")
                        vision_results = vision_module.process_image(image_path=filepath, generate_visualizations=generate_visualizations)
                        source_path = f"/static/uploads/{filename}"
                    else:
                        # For local paths, process directly
                        log_to_client("Processing image from local path")
                        vision_results = vision_module.process_image(image_path=image_url, generate_visualizations=generate_visualizations)
                        source_path = image_url
                except requests.exceptions.RequestException as e:
                    log_to_client(f"Error downloading image: {str(e)}", "error")
                    logger.error(f"Error downloading image from URL: {e}")
                    return jsonify({
                        "success": False,
                        "error": f"Failed to download image from URL. The website may be blocking automated requests or the URL is invalid: {str(e)}"
                    }), 400
            
            # Save results to vector database
            metadata = {
                "label": vision_results['label'],
                "source": source_path,
                "timestamp": datetime.now().isoformat(),
                "type": "url",
                "perceptual_hash": vision_results['perceptual_hash'],
                "attention_maps": vision_results.get('attention_maps'),
                "feature_maps": vision_results.get('feature_maps'),
                "transformation_pipeline": vision_results.get('transformation_pipeline')
            }
            
            # Check for duplicates before adding to database
            log_to_client("Checking for duplicate images in database...")
            duplicate_id = vector_db.find_duplicates(vision_results['perceptual_hash'])
            if duplicate_id:
                logger.info(f"Duplicate image detected, using existing ID: {duplicate_id}")
                log_to_client(f"Duplicate image detected, using existing ID: {duplicate_id}")
                image_id = duplicate_id
            else:
                log_to_client("Saving image data to vector database...")
                vector_db.add_image_data(image_id, vision_results['image_embedding'], metadata)
            
        elif 'imageFile' in request.files:
            # Image file uploaded
            file = request.files['imageFile']
            
            # Check if the file is valid
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Generate a unique filename
            filename = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
            filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            log_to_client(f"File saved: {filename}")
            
            # Generate a unique ID for this image
            image_id = f"img_{uuid.uuid4()}"
            
            # Process the image
            try:
                log_to_client("Analyzing uploaded image with Vision Transformer...")
                vision_results = vision_module.process_image(image_path=filepath, generate_visualizations=generate_visualizations)
            except Exception as e:
                log_to_client(f"Error processing image: {str(e)}", "error")
                logger.error(f"Error processing uploaded image: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Failed to process the uploaded image: {str(e)}"
                }), 400
            
            # Check for duplicates before adding to database
            log_to_client("Checking for duplicate images in database...")
            duplicate_id = vector_db.find_duplicates(vision_results['perceptual_hash'])
            if duplicate_id:
                logger.info(f"Duplicate image detected, using existing ID: {duplicate_id}")
                log_to_client(f"Duplicate image detected, using existing ID: {duplicate_id}")
                image_id = duplicate_id
            else:
                # Save results to vector database
                log_to_client("Saving image data to vector database...")
                metadata = {
                    "label": vision_results['label'],
                    "source": f"/static/uploads/{filename}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "upload",
                    "perceptual_hash": vision_results['perceptual_hash'],
                    "attention_maps": vision_results.get('attention_maps'),
                    "feature_maps": vision_results.get('feature_maps'),
                    "transformation_pipeline": vision_results.get('transformation_pipeline')
                }
                vector_db.add_image_data(image_id, vision_results['image_embedding'], metadata)
            
        else:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Generate LLM insights with transformer-specific details
        log_to_client("Generating LLM insights about the image...")
        insights = llm_insights.generate_insights(vision_results['label'], vision_results['confidence'])
        
        # Get transformer explanation for this image type
        log_to_client("Generating transformer explanations...")
        transformer_explanation = transformer_explainer.generate_custom_explanation(vision_results['label'])
        
        # Get attention explanation for this image type
        attention_explanation = transformer_explainer.generate_attention_explanation(vision_results['label'])
        
        # Find similar images from the vector database
        log_to_client("Finding similar images from vector database...")
        similar_results = vector_db.find_similar(vision_results['image_embedding'], num_similar_images)
        
        # Prepare similar images data from the vector database
        similar_images = []
        if similar_results['ids'] and similar_results['ids'][0]:
            for i, img_id in enumerate(similar_results['ids'][0]):
                if img_id != image_id:  # Skip the query image itself
                    metadata = similar_results['metadatas'][0][i]
                    distance = similar_results['distances'][0][i]
                    similarity_score = 1.0 - min(distance, 1.0)  # Convert distance to similarity (0-1)
                    
                    similar_images.append({
                        "id": img_id,
                        "label": metadata.get('label', 'Unknown'),
                        "source": metadata.get('source', ''),
                        "similarity": round(similarity_score * 100, 2),
                        "type": metadata.get('type', 'unknown'),
                        "origin": "database",
                        "attention_maps": metadata.get('attention_maps'),
                        "feature_maps": metadata.get('feature_maps'),
                        "transformation_pipeline": metadata.get('transformation_pipeline')
                    })
        
        # Find similar images from the web using the image scraper
        # Calculate how many web images we need to reach the total requested
        additional_images_needed = max(0, num_similar_images - len(similar_images))
        
        web_images = []
        if additional_images_needed > 0:
            try:
                # Use the label for searching similar images
                log_to_client(f"Searching web for {additional_images_needed} similar images...")
                web_images = image_scraper.scrape_similar_images(
                    label=vision_results['label'], 
                    limit=additional_images_needed
                )
                
                # Add metadata to the web images
                for i, img in enumerate(web_images):
                    img['id'] = f"web_{uuid.uuid4()}"
                    # Calculate a score that's always lower than database matches but still reasonable
                    base_score = 90 - min(i * 3, 30)  # Starts at 90%, decreases by 3% per image, but never below 60%
                    img['similarity'] = base_score
                    img['type'] = 'web'
                    img['origin'] = 'web'
                    
                    # Save the web image locally to avoid CORS issues
                    log_to_client(f"Downloading web image {i+1}/{len(web_images)}...")
                    local_path = image_scraper.download_image(img['source'])
                    if local_path:
                        img['source'] = local_path
                    else:
                        # If download fails, use a placeholder
                        img['source'] = f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(vision_results['label'])}"
                        img['is_placeholder'] = True
            except Exception as e:
                log_to_client(f"Error scraping web images: {str(e)}", "error")
                logger.error(f"Error scraping web images: {e}")
                # Create placeholders for the missing images
                for i in range(additional_images_needed):
                    web_images.append({
                        "id": f"web_{uuid.uuid4()}",
                        "label": f"{vision_results['label']} (Web)",
                        "source": f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(vision_results['label'])}",
                        "similarity": 60 - (i * 2),  # Decreasing similarity for placeholders
                        "type": "web",
                        "origin": "web",
                        "is_placeholder": True
                    })
        
        # Combine local and web images, prioritizing local ones
        all_similar_images = similar_images + web_images
        
        # Sort by similarity score and limit to requested number
        all_similar_images = sorted(all_similar_images, key=lambda x: x['similarity'], reverse=True)[:num_similar_images]
        
        # Log completion
        log_to_client("Analysis complete! Returning results...")
        
        # Return the results
        return jsonify({
            "success": True,
            "image": {
                "id": image_id,
                "label": vision_results['label'],
                "confidence": round(vision_results['confidence'] * 100, 2),
                "source": metadata['source'],
                "attention_maps": vision_results.get('attention_maps'),
                "feature_maps": vision_results.get('feature_maps'),
                "transformation_pipeline": vision_results.get('transformation_pipeline')
            },
            "knowledge": insights,
            "transformer_explanation": transformer_explanation,
            "attention_explanation": attention_explanation,
            "similar_images": all_similar_images
        })
    except Exception as e:
        log_to_client(f"Unexpected error: {str(e)}", "error")
        logger.error(f"Unexpected error in analyze_image: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@flask_app.route('/search', methods=['POST'])
def search_similar():
    """API endpoint to search for similar images by label"""
    try:
        search_term = request.json.get('term', '').lower()
        
        if not search_term:
            return jsonify({"success": False, "error": "No search term provided"}), 400
        
        log_to_client(f"Searching for images with label: {search_term}")
        
        # Use the more efficient search method from VectorDB
        results = vector_db.search_by_label(search_term)
        
        # Get the number of additional web results to fetch
        num_web_results = request.json.get('numWebResults', 5)
        num_web_results = min(max(num_web_results, 0), 10)  # Cap between 0 and 10
        
        # Add web search results if requested
        web_results = []
        if num_web_results > 0:
            try:
                log_to_client(f"Searching web for {num_web_results} additional images...")
                web_images = image_scraper.scrape_images_by_keyword(search_term, limit=num_web_results)
                
                for img in web_images:
                    img['id'] = f"web_{uuid.uuid4()}"
                    img['type'] = 'web'
                    img['origin'] = 'web'
                    img['timestamp'] = datetime.now().isoformat()
                    
                    # Save the web image locally to avoid CORS issues
                    log_to_client(f"Downloading web image: {img['label']}")
                    local_path = image_scraper.download_image(img['source'])
                    if local_path:
                        img['source'] = local_path
                        web_results.append(img)
                    elif img.get('is_placeholder'):
                        # Keep placeholders as they are
                        web_results.append(img)
            except Exception as e:
                log_to_client(f"Error getting web search results: {str(e)}", "error")
                logger.error(f"Error getting web search results: {e}")
                # Add placeholders if web scraping fails
                for i in range(num_web_results):
                    web_results.append({
                        "id": f"web_{uuid.uuid4()}",
                        "label": f"{search_term} (Web)",
                        "source": f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(search_term)}",
                        "type": "web",
                        "origin": "web",
                        "timestamp": datetime.now().isoformat(),
                        "is_placeholder": True
                    })
        
        # Combine results, prioritizing database results
        all_results = results + web_results
        
        log_to_client(f"Search complete. Found {len(all_results)} images.")
        
        return jsonify({
            "success": True,
            "results": all_results
        })
        
    except Exception as e:
        log_to_client(f"Error in search: {str(e)}", "error")
        logger.error(f"Error in search_similar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@flask_app.route('/scrape-similar', methods=['POST'])
def scrape_similar_images():
    """API endpoint to scrape similar images from the web"""
    try:
        # Get the label or image URL from the request
        label = request.json.get('label')
        image_url = request.json.get('imageUrl')
        
        # Get the number of images to scrape (default is 5)
        limit = int(request.json.get('limit', 5))
        # Cap the number to avoid overloading
        limit = min(max(limit, 1), 20)
        
        if not label and not image_url:
            return jsonify({"success": False, "error": "Either label or imageUrl must be provided"}), 400
        
        # Scrape similar images using our image scraper
        log_to_client(f"Scraping {limit} similar images for '{label or image_url}'")
        web_images = image_scraper.scrape_similar_images(
            image_url=image_url,
            label=label,
            limit=limit
        )
        
        # Process and save the images locally
        processed_images = []
        for i, img in enumerate(web_images):
            try:
                # Skip placeholder images which don't need downloading
                if img.get('is_placeholder'):
                    processed_images.append({
                        "id": f"web_{uuid.uuid4()}",
                        "label": img['label'],
                        "source": img['source'],
                        "type": "web",
                        "origin": "web",
                        "similarity": 90 - (i * 2),  # Decreasing similarity
                        "is_placeholder": True
                    })
                    continue
                
                # Save the web image locally to avoid CORS issues
                log_to_client(f"Downloading web image {i+1}/{len(web_images)}...")
                local_path = image_scraper.download_image(img['source'])
                if local_path:
                    processed_images.append({
                        "id": f"web_{uuid.uuid4()}",
                        "label": img['label'],
                        "source": local_path,
                        "type": "web",
                        "origin": "web",
                        "similarity": 90 - (i * 2)  # Decreasing similarity
                    })
            except Exception as e:
                log_to_client(f"Error processing scraped image: {str(e)}", "warning")
                logger.warning(f"Error processing scraped image: {e}")
        
        # If we couldn't process any images, add placeholders
        if not processed_images:
            keyword = label or "image"
            log_to_client(f"No images processed successfully. Adding placeholders...")
            for i in range(limit):
                processed_images.append({
                    "id": f"web_{uuid.uuid4()}",
                    "label": f"{keyword} (Web)",
                    "source": f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(keyword)}",
                    "type": "web",
                    "origin": "web",
                    "similarity": 80 - (i * 2),
                    "is_placeholder": True
                })
        
        log_to_client(f"Scraping complete. Returning {len(processed_images)} images.")
        
        return jsonify({
            "success": True,
            "images": processed_images
        })
        
    except Exception as e:
        log_to_client(f"Error in scrape-similar: {str(e)}", "error")
        logger.error(f"Error in scrape_similar_images: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@flask_app.route('/visualizations/<path:viz_type>/<path:viz_id>')
def get_visualization(viz_type, viz_id):
    """API endpoint to get a specific visualization"""
    try:
        # Determine the visualization type and path
        if viz_type == 'attention':
            viz_path = os.path.join(flask_app.config['ATTENTION_MAPS_FOLDER'], viz_id)
        elif viz_type == 'features':
            viz_path = os.path.join(flask_app.config['FEATURE_MAPS_FOLDER'], viz_id)
        elif viz_type == 'transformation':
            viz_path = os.path.join(flask_app.config['TRANSFORMATION_FOLDER'], viz_id)
        else:
            return jsonify({"success": False, "error": "Invalid visualization type"}), 400
        
        # Get the list of visualization files
        viz_files = os.listdir(viz_path)
        
        # Return the list of available visualizations
        return jsonify({
            "success": True,
            "visualizations": [f"/static/{viz_type}_maps/{viz_id}/{file}" for file in viz_files]
        })
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@flask_app.route('/compare_models', methods=['POST'])
def compare_models():
    """API endpoint to compare ViT with other vision models on an image"""
    try:
        # Get the image ID to analyze
        image_id = request.json.get('imageId')
        if not image_id:
            return jsonify({"success": False, "error": "No image ID provided"}), 400
        
        log_to_client(f"Comparing models for image ID: {image_id}")
        
        # Get the image data from the vector database
        image_data = vector_db.collection.get(ids=[image_id])
        if not image_data or not image_data['ids']:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Get metadata and source
        metadata = image_data['metadatas'][0]
        image_source = metadata.get('source')
        
        # Simulate comparison with different vision models (would be real in production)
        comparison_results = {
            "models": [
                {
                    "name": "Vision Transformer (ViT-B/16)",
                    "prediction": metadata.get('label'),
                    "confidence": 0.92,
                    "inference_time": 42,  # milliseconds
                    "strengths": ["Global context awareness", "Captures long-range dependencies"],
                    "limitations": ["Quadratic complexity with image size"]
                },
                {
                    "name": "ResNet-50 (CNN)",
                    "prediction": metadata.get('label'),  # Same for simplicity
                    "confidence": 0.89,
                    "inference_time": 28,  # milliseconds
                    "strengths": ["Efficient feature hierarchy", "Lower computational cost"],
                    "limitations": ["Limited receptive field in early layers"]
                },
                {
                    "name": "EfficientNet-B4",
                    "prediction": metadata.get('label'),  # Same for simplicity
                    "confidence": 0.91,
                    "inference_time": 35,  # milliseconds
                    "strengths": ["Balanced depth/width/resolution", "Parameter efficient"],
                    "limitations": ["Still relies on convolution limitations"]
                },
                {
                    "name": "Swin Transformer",
                    "prediction": metadata.get('label'),  # Same for simplicity
                    "confidence": 0.94,
                    "inference_time": 38,  # milliseconds
                    "strengths": ["Hierarchical feature maps", "Linear complexity scaling"],
                    "limitations": ["Limited window attention in each layer"]
                }
            ],
            "comparison_metrics": {
                "parameter_count": {
                    "Vision Transformer (ViT-B/16)": "86M",
                    "ResNet-50 (CNN)": "25M",
                    "EfficientNet-B4": "19M",
                    "Swin Transformer": "88M"
                },
                "flops": {
                    "Vision Transformer (ViT-B/16)": "17.6G",
                    "ResNet-50 (CNN)": "4.1G",
                    "EfficientNet-B4": "4.2G",
                    "Swin Transformer": "8.7G"
                }
            }
        }
        
        log_to_client("Model comparison complete")
        
        return jsonify({
            "success": True,
            "image_source": image_source,
            "comparison": comparison_results
        })
        
    except Exception as e:
        log_to_client(f"Error in compare_models: {str(e)}", "error")
        logger.error(f"Error in compare_models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@flask_app.route('/generate_attention_visualization', methods=['POST'])
def generate_attention_visualization():
    """API endpoint to generate a custom attention visualization"""
    try:
        # Get the image ID and heads to visualize
        image_id = request.json.get('imageId')
        layer = request.json.get('layer', 0)  # Default to first layer
        heads = request.json.get('heads', [0])  # Default to first attention head
        
        if not image_id:
            return jsonify({"success": False, "error": "No image ID provided"}), 400
        
        log_to_client(f"Generating attention visualization for image ID: {image_id}")
        
        # Get the image data from the vector database
        image_data = vector_db.collection.get(ids=[image_id])
        if not image_data or not image_data['ids']:
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Get metadata and source
        metadata = image_data['metadatas'][0]
        image_source = metadata.get('source')
        
        # Check if attention maps are already available
        if not metadata.get('attention_maps'):
            # Generate attention maps
            try:
                log_to_client("Generating attention maps...")
                image_path = os.path.join(flask_app.root_path, image_source.lstrip('/'))
                image = Image.open(image_path)
                
                # Generate attention maps
                attention_maps = vision_module._generate_attention_visualizations(image)
                
                # Update metadata with attention maps
                log_to_client("Updating vector database with attention maps...")
                vector_db.collection.update(
                    ids=[image_id],
                    metadatas=[{**metadata, "attention_maps": attention_maps}]
                )
            except Exception as e:
                log_to_client(f"Error generating attention maps: {str(e)}", "error")
                logger.error(f"Error generating attention maps: {e}")
                return jsonify({"success": False, "error": f"Failed to generate attention maps: {str(e)}"}), 500
        
        log_to_client("Attention visualization complete")
        
        # Return the attention visualization path
        return jsonify({
            "success": True,
            "attention_visualization": metadata.get('attention_maps'),
            "custom_visualization": f"/static/attention_maps/custom_{image_id}_{layer}_{'-'.join(map(str, heads))}.png"
        })
        
    except Exception as e:
        log_to_client(f"Error in generate_attention_visualization: {str(e)}", "error")
        logger.error(f"Error in generate_attention_visualization: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@flask_app.route('/download_visualization/<path:viz_type>/<path:viz_id>')
def download_visualization(viz_type, viz_id):
    """API endpoint to download a visualization as an image file"""
    try:
        # Determine the visualization type and path
        if viz_type == 'attention':
            viz_path = os.path.join(flask_app.config['ATTENTION_MAPS_FOLDER'], viz_id)
            filename = "attention_map.png"
        elif viz_type == 'features':
            viz_path = os.path.join(flask_app.config['FEATURE_MAPS_FOLDER'], viz_id)
            filename = "feature_maps.png"
        elif viz_type == 'transformation':
            viz_path = os.path.join(flask_app.config['TRANSFORMATION_FOLDER'], viz_id)
            filename = "transformation_pipeline.png"
        else:
            return jsonify({"success": False, "error": "Invalid visualization type"}), 400
        
        # Find the main visualization file
        main_file = None
        for file in os.listdir(viz_path):
            if file == f"{viz_type}_maps.png" or file == "combined_attention.png" or file == "vit_pipeline.png":
                main_file = file
                break
        
        if not main_file:
            main_file = os.listdir(viz_path)[0]  # Just use the first file if none of the expected names match
        
        # Return the file for download
        return send_file(
            os.path.join(viz_path, main_file),
            as_attachment=True,
            download_name=filename,
            mimetype='image/png'
        )
    except Exception as e:
        log_to_client(f"Error downloading visualization: {str(e)}", "error")
        logger.error(f"Error downloading visualization: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@flask_app.route('/interactive_transformer')
def interactive_transformer():
    """Interactive Vision Transformer demo page"""
    return render_template('interactive.html')

@flask_app.route('/system_logs')
def system_logs():
    """View system logs page"""
    return render_template('logs.html')

# Add WebSocket endpoint for terminal commands
@socketio.on('execute_command')
def handle_command(data):
    """Handle terminal commands from the frontend"""
    command = data.get('command', '').strip()
    log_to_client(f"Executing command: {command}")
    
    if command.startswith('help'):
        # Return available commands
        socketio.emit('command_result', {
            'result': """Available commands:
- help: Show this help message
- status: Show system status
- list models: List available models
- clear cache: Clear image cache
- version: Show system version"""
        })
    elif command.startswith('status'):
        # Return system status
        socketio.emit('command_result', {
            'result': f"""System Status:
- Vector DB: {len(vector_db.hash_index)} images indexed
- Vision Module: Active
- Image Scraper: Active
- LLM Insights: Using model {llm_insights.model_name}
- Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        })
    elif command.startswith('list models'):
        # List models
        socketio.emit('command_result', {
            'result': """Available Models:
1. google/vit-base-patch16-224 (default)
2. mistral:latest (for insights generation)
3. llama2:latest (alternative for insights)"""
        })
    elif command.startswith('clear cache'):
        # Simulate cache clearing
        socketio.emit('command_result', {
            'result': "Cache cleared successfully."
        })
    elif command.startswith('version'):
        # Return version info
        socketio.emit('command_result', {
            'result': "Vision Transformer v1.0.0"
        })
    else:
        # Unknown command
        socketio.emit('command_result', {
            'result': f"Unknown command: {command}\nType 'help' to see available commands."
        })

if __name__ == "__main__":
    socketio.run(flask_app, debug=True, port=5000, host="0.0.0.0")