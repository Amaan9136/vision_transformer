"""
Vision Knowledge Explorer - Flask Web Application

This application combines Vision Transformers, Vector Databases, and Knowledge Graphs
to create a system that can analyze images from the web, find similar images,
and provide insightful descriptions with a modern UI.
"""

import os
import logging
import torch
import json
import uuid
import requests
import base64
import re
from PIL import Image
from io import BytesIO
from datetime import datetime
from transformers import ViTForImageClassification, ViTImageProcessor
import chromadb
from flask import Flask, render_template, request, jsonify
import urllib.parse
from bs4 import BeautifulSoup
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User agent for scraping
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# ---- 1. Vision Transformer Setup ----

class VisionModule:
    def __init__(self):
        # Load pre-trained Vision Transformer model
        logger.info("Initializing Vision Transformer model...")
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        logger.info("Vision Transformer model loaded successfully!")
    
    def process_image(self, image_path=None, image_data=None):
        """Process an image and return detected objects/concepts"""
        logger.info(f"Processing image: {image_path if image_path else 'from data'}")
        try:
            if image_path:
                if image_path.startswith('http'):
                    # Load image from URL
                    response = requests.get(image_path, timeout=10)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    image_data = response.content
                    image = Image.open(BytesIO(image_data))
                else:
                    # Load image from local path
                    image = Image.open(image_path)
            elif image_data:
                # Load image from binary data
                image = Image.open(BytesIO(image_data))
            else:
                raise ValueError("Either image_path or image_data must be provided")
            
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get the predicted class ID
            predicted_class_id = logits.argmax(-1).item()
            
            # Get the predicted class label
            label = self.model.config.id2label[predicted_class_id]
            
            # Create embedding from the logits
            embedding = logits[0].tolist()
            
            # Return the detected object/concept
            return {
                "label": label,
                "confidence": float(logits.softmax(dim=-1)[0][predicted_class_id].item()),
                "image_embedding": embedding
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

# ---- 2. Vector Database Setup ----

class VectorDB:
    def __init__(self):
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.Client()
        
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name="image_embeddings")
            logger.info("Using existing ChromaDB collection")
        except:
            # Create a new collection if it doesn't exist
            self.collection = self.client.create_collection(name="image_embeddings")
            logger.info("Created new ChromaDB collection")
    
    def add_image_data(self, image_id, embedding, metadata=None):
        """Add image embedding to vector database"""
        try:
            self.collection.add(
                ids=[image_id],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            logger.error(f"Error adding to vector DB: {e}")
            return False
    
    def find_similar(self, embedding, limit=9):
        """Find similar images based on embedding"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

# ---- 3. Simplified Knowledge Base ----

class SimpleKnowledgeBase:
    """A simplified knowledge base with common objects and concepts"""
    
    def __init__(self):
        # Initialize with a dictionary of concepts and their properties
        self.knowledge = {
            "car": {
                "type": "vehicle",
                "powered_by": "engine",
                "description": "A wheeled motor vehicle used for transportation.",
                "relations": [
                    {"relationship": "HAS_PART", "related_concept": "wheel", "related_labels": ["Component"]},
                    {"relationship": "HAS_PART", "related_concept": "engine", "related_labels": ["Component"]},
                    {"relationship": "IS_A", "related_concept": "vehicle", "related_labels": ["Category"]}
                ]
            },
            "tesla": {
                "type": "electric vehicle",
                "manufacturer": "Tesla, Inc.",
                "founder": "Elon Musk",
                "description": "An electric vehicle manufactured by Tesla, Inc.",
                "relations": [
                    {"relationship": "IS_A", "related_concept": "car", "related_labels": ["Category"]},
                    {"relationship": "USES", "related_concept": "electricity", "related_labels": ["Energy"]},
                    {"relationship": "MANUFACTURED_BY", "related_concept": "Tesla, Inc.", "related_labels": ["Company"]}
                ]
            },
            "smartphone": {
                "type": "electronic device",
                "purpose": "communication",
                "description": "A portable device that combines mobile telephone and computing functions.",
                "relations": [
                    {"relationship": "HAS_COMPONENT", "related_concept": "screen", "related_labels": ["Component"]},
                    {"relationship": "HAS_COMPONENT", "related_concept": "camera", "related_labels": ["Component"]},
                    {"relationship": "HAS_FUNCTION", "related_concept": "communication", "related_labels": ["Function"]}
                ]
            },
            "laptop": {
                "type": "electronic device",
                "purpose": "computing",
                "description": "A portable personal computer with a clamshell form factor.",
                "relations": [
                    {"relationship": "HAS_COMPONENT", "related_concept": "keyboard", "related_labels": ["Component"]},
                    {"relationship": "HAS_COMPONENT", "related_concept": "screen", "related_labels": ["Component"]},
                    {"relationship": "HAS_FUNCTION", "related_concept": "computing", "related_labels": ["Function"]}
                ]
            },
            "tree": {
                "type": "plant",
                "category": "nature",
                "description": "A perennial plant with an elongated stem, or trunk, supporting branches and leaves.",
                "relations": [
                    {"relationship": "HAS_PART", "related_concept": "leaf", "related_labels": ["Component"]},
                    {"relationship": "HAS_PART", "related_concept": "trunk", "related_labels": ["Component"]},
                    {"relationship": "IS_A", "related_concept": "plant", "related_labels": ["Category"]}
                ]
            },
            "dog": {
                "type": "animal",
                "category": "pet",
                "description": "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking, howling, or whining voice.",
                "relations": [
                    {"relationship": "IS_A", "related_concept": "mammal", "related_labels": ["Category"]},
                    {"relationship": "HAS_PART", "related_concept": "tail", "related_labels": ["Component"]},
                    {"relationship": "BELONGS_TO", "related_concept": "canidae", "related_labels": ["Family"]}
                ]
            }
        }
    
    def query_knowledge(self, concept):
        """Query knowledge base for information about a concept"""
        # Normalize the concept (lowercase) for matching
        concept = concept.lower()
        
        # Check for exact match
        if concept in self.knowledge:
            return self.knowledge[concept]
        
        # Check for partial matches
        for key in self.knowledge:
            if key in concept or concept in key:
                return self.knowledge[key]
        
        # No match found
        return {}

# ---- 4. Web Scraper for Similar Images ----

class ImageScraper:
    """Scrapes similar images from the web based on keywords or an image"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': USER_AGENT
        }
    
    def scrape_images_by_keyword(self, keyword, limit=5):
        """Scrape images based on a keyword search"""
        try:
            logger.info(f"Scraping images for keyword: {keyword}")
            
            # Format the search query
            search_query = urllib.parse.quote(f"{keyword}")
            search_url = f"https://www.bing.com/images/search?q={search_query}&form=HDRSC2&first=1"
            
            response = requests.get(search_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs
            images = []
            image_elements = soup.select('.mimg')
            
            for element in image_elements:
                if len(images) >= limit:
                    break
                    
                img_src = element.get('src') or element.get('data-src')
                if img_src and img_src.startswith('http'):
                    # Validate image URL
                    try:
                        # Attempt to get image headers to check if it's a valid image
                        img_head = requests.head(img_src, headers=self.headers, timeout=5)
                        content_type = img_head.headers.get('content-type', '')
                        
                        if 'image' in content_type:
                            images.append({
                                'source': img_src,
                                'label': keyword
                            })
                    except Exception as e:
                        logger.warning(f"Skipping invalid image URL: {e}")
                        continue
            
            # If we don't have enough images yet, try a different selector
            if len(images) < limit:
                img_tags = soup.select('img.mimg')
                for img in img_tags:
                    if len(images) >= limit:
                        break
                        
                    img_src = img.get('src') or img.get('data-src')
                    if img_src and img_src.startswith('http') and img_src not in [img['source'] for img in images]:
                        # Validate image URL
                        try:
                            img_head = requests.head(img_src, headers=self.headers, timeout=5)
                            content_type = img_head.headers.get('content-type', '')
                            
                            if 'image' in content_type:
                                images.append({
                                    'source': img_src,
                                    'label': keyword
                                })
                        except Exception as e:
                            logger.warning(f"Skipping invalid image URL: {e}")
                            continue
            
            logger.info(f"Found {len(images)} images for keyword: {keyword}")
            return images
            
        except Exception as e:
            logger.error(f"Error scraping images for keyword {keyword}: {e}")
            return []
    
    def scrape_similar_images(self, image_url=None, label=None, limit=5):
        """Find similar images based on an existing image or a label"""
        # If we have a label, use it for keyword search
        if label:
            return self.scrape_images_by_keyword(label, limit)
        elif image_url:
            # For image-based search, we'll use the image URL to extract a label first
            # This is a simplification; actual reverse image search would be more complex
            vision_results = vision_module.process_image(image_path=image_url)
            label = vision_results['label']
            return self.scrape_images_by_keyword(label, limit)
        else:
            logger.error("Either image_url or label must be provided")
            return []
    
    def scrape_images_concurrent(self, keywords, limit_per_keyword=1):
        """Scrape images for multiple keywords concurrently"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Create a dictionary mapping futures to their corresponding keywords
            future_to_keyword = {
                executor.submit(self.scrape_images_by_keyword, keyword, limit_per_keyword): keyword
                for keyword in keywords
            }
            
            for future in concurrent.futures.as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    images = future.result()
                    results.extend(images)
                    logger.info(f"Collected {len(images)} images for keyword: {keyword}")
                except Exception as e:
                    logger.error(f"Error collecting images for keyword {keyword}: {e}")
        
        return results
    
    def download_image(self, image_url):
        """Download an image from a URL and save it to the uploads folder"""
        try:
            response = requests.get(image_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Create a unique filename
            file_extension = self._get_file_extension(response.headers.get('content-type', 'image/jpeg'))
            filename = f"scraped_{uuid.uuid4()}{file_extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return f"/static/uploads/{filename}"
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")
            return None
    
    def _get_file_extension(self, content_type):
        """Get the file extension based on content type"""
        if 'jpeg' in content_type or 'jpg' in content_type:
            return '.jpg'
        elif 'png' in content_type:
            return '.png'
        elif 'gif' in content_type:
            return '.gif'
        elif 'webp' in content_type:
            return '.webp'
        else:
            return '.jpg'  # Default to jpg

# Initialize modules globally for the Flask app
vision_module = VisionModule()
vector_db = VectorDB()
knowledge_base = SimpleKnowledgeBase()
image_scraper = ImageScraper()

# ---- 5. Flask Routes ----

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

def is_base64_image(data_url):
    """Check if a string is a base64 encoded image URL"""
    return isinstance(data_url, str) and data_url.startswith('data:image/')

def decode_base64_image(data_url):
    """Decode a base64 encoded image URL to binary data"""
    try:
        # Extract the base64 part
        pattern = r'data:image/[^;]+;base64,(.+)'
        match = re.match(pattern, data_url)
        if match:
            base64_data = match.group(1)
            # Decode base64 data
            return base64.b64decode(base64_data)
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
    return None

def sanitize_filename(filename):
    """Sanitize a filename to ensure it's safe for filesystem storage"""
    # Remove any path separators and limit length
    basename = os.path.basename(filename)
    # Remove any potentially dangerous characters
    safe_name = re.sub(r'[^\w\-\.]', '_', basename)
    # Limit length to avoid "filename too long" errors
    if len(safe_name) > 50:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:46] + ext if ext else name[:50]
    return safe_name

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """API endpoint to analyze an image"""
    try:
        if 'imageUrl' in request.form:
            # Image URL provided
            image_url = request.form['imageUrl']
            
            # Generate a unique ID for this image
            image_id = f"img_{uuid.uuid4()}"
            
            # Check if this is a base64 encoded image
            if is_base64_image(image_url):
                # Decode base64 data
                image_data = decode_base64_image(image_url)
                if not image_data:
                    return jsonify({"error": "Invalid base64 image data"}), 400
                
                # Save base64 image to a file for record keeping
                filename = f"{uuid.uuid4()}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                # Process the image data directly
                vision_results = vision_module.process_image(image_data=image_data)
                
                # Use the file path as source for the record
                source_path = f"/static/uploads/{filename}"
            else:
                try:
                    # For URLs, download the image first to handle potential errors better
                    if image_url.startswith('http'):
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        
                        # Get a safe filename from the URL
                        url_filename = os.path.basename(image_url.split('?')[0])  # Remove query parameters
                        safe_filename = sanitize_filename(url_filename)
                        if not safe_filename or safe_filename == '':
                            safe_filename = 'image.jpg'
                            
                        # Create a unique filename to avoid collisions
                        filename = f"{uuid.uuid4()}_{safe_filename}"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        # Save the image to a file
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        # Process the image from the saved file
                        vision_results = vision_module.process_image(image_path=filepath)
                        source_path = f"/static/uploads/{filename}"
                    else:
                        # For local paths, process directly
                        vision_results = vision_module.process_image(image_path=image_url)
                        source_path = image_url
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading image from URL: {e}")
                    return jsonify({"error": f"Failed to download image from URL: {str(e)}"}), 400
            
            # Save results to vector database
            metadata = {
                "label": vision_results['label'],
                "source": source_path,
                "timestamp": datetime.now().isoformat(),
                "type": "url"
            }
            vector_db.add_image_data(image_id, vision_results['image_embedding'], metadata)
            
        elif 'imageFile' in request.files:
            # Image file uploaded
            file = request.files['imageFile']
            
            # Check if the file is valid
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Generate a unique filename
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Generate a unique ID for this image
            image_id = f"img_{uuid.uuid4()}"
            
            # Process the image
            vision_results = vision_module.process_image(image_path=filepath)
            
            # Save results to vector database
            metadata = {
                "label": vision_results['label'],
                "source": f"/static/uploads/{filename}",
                "timestamp": datetime.now().isoformat(),
                "type": "upload"
            }
            vector_db.add_image_data(image_id, vision_results['image_embedding'], metadata)
            
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Query knowledge base
        knowledge_info = knowledge_base.query_knowledge(vision_results['label'])
        
        # Find similar images from the vector database
        similar_results = vector_db.find_similar(vision_results['image_embedding'])
        
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
                        "origin": "database"
                    })
        
        # Find similar images from the web using the image scraper
        web_images = []
        try:
            # Use the label for searching similar images
            web_images = image_scraper.scrape_similar_images(label=vision_results['label'], limit=5)
            
            # Add metadata to the web images
            for i, img in enumerate(web_images):
                img['id'] = f"web_{uuid.uuid4()}"
                img['similarity'] = 100 - (i * 5)  # Mock similarity score decreasing by 5% for each result
                img['type'] = 'web'
                img['origin'] = 'web'
                
                # Save the web image locally to avoid CORS issues
                local_path = image_scraper.download_image(img['source'])
                if local_path:
                    img['source'] = local_path
        except Exception as e:
            logger.error(f"Error scraping web images: {e}")
        
        # Combine local and web images, prioritizing local ones
        all_similar_images = similar_images + web_images
        
        # Sort by similarity score and limit to 9 results
        all_similar_images = sorted(all_similar_images, key=lambda x: x['similarity'], reverse=True)[:9]
        
        # Return the results
        return jsonify({
            "success": True,
            "image": {
                "id": image_id,
                "label": vision_results['label'],
                "confidence": round(vision_results['confidence'] * 100, 2),
                "source": metadata['source']
            },
            "knowledge": knowledge_info,
            "similar_images": all_similar_images
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search_similar():
    """API endpoint to search for similar images by label"""
    try:
        search_term = request.json.get('term', '').lower()
        
        if not search_term:
            return jsonify({"error": "No search term provided"}), 400
        
        # For simplicity, we're using a brute force approach
        # In a production app, you'd want to use a more efficient search mechanism
        
        # Get all items from the collection
        collection_items = vector_db.collection.get()
        
        results = []
        if collection_items['ids']:
            for i, item_id in enumerate(collection_items['ids']):
                metadata = collection_items['metadatas'][i]
                label = metadata.get('label', '').lower()
                
                if search_term in label:
                    results.append({
                        "id": item_id,
                        "label": metadata.get('label', 'Unknown'),
                        "source": metadata.get('source', ''),
                        "type": metadata.get('type', 'unknown'),
                        "timestamp": metadata.get('timestamp', ''),
                        "origin": "database"
                    })
        
        # Add web search results
        web_results = []
        try:
            web_images = image_scraper.scrape_images_by_keyword(search_term, limit=5)
            
            for img in web_images:
                img['id'] = f"web_{uuid.uuid4()}"
                img['type'] = 'web'
                img['origin'] = 'web'
                img['timestamp'] = datetime.now().isoformat()
                
                # Save the web image locally to avoid CORS issues
                local_path = image_scraper.download_image(img['source'])
                if local_path:
                    img['source'] = local_path
                    web_results.append(img)
        except Exception as e:
            logger.error(f"Error getting web search results: {e}")
        
        # Combine results, prioritizing database results
        all_results = results + web_results
        
        return jsonify({
            "success": True,
            "results": all_results
        })
        
    except Exception as e:
        logger.error(f"Error in search_similar: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/scrape-similar', methods=['POST'])
def scrape_similar_images():
    """API endpoint to scrape similar images from the web"""
    try:
        # Get the label or image URL from the request
        label = request.json.get('label')
        image_url = request.json.get('imageUrl')
        
        if not label and not image_url:
            return jsonify({"error": "Either label or imageUrl must be provided"}), 400
        
        # Scrape similar images using our image scraper
        web_images = image_scraper.scrape_similar_images(
            image_url=image_url,
            label=label,
            limit=5
        )
        
        # Process and save the images locally
        processed_images = []
        for img in web_images:
            try:
                # Save the web image locally to avoid CORS issues
                local_path = image_scraper.download_image(img['source'])
                if local_path:
                    processed_images.append({
                        "id": f"web_{uuid.uuid4()}",
                        "label": img['label'],
                        "source": local_path,
                        "type": "web",
                        "origin": "web",
                        "similarity": 95  # Mock similarity score
                    })
            except Exception as e:
                logger.warning(f"Error processing scraped image: {e}")
        
        return jsonify({
            "success": True,
            "images": processed_images
        })
        
    except Exception as e:
        logger.error(f"Error in scrape_similar_images: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")