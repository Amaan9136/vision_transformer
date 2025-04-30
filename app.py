import os
import logging
import torch
import json
import uuid
import requests
import base64
import re
import random
from PIL import Image
from io import BytesIO
from datetime import datetime
from transformers import ViTForImageClassification, ViTImageProcessor
import chromadb
from flask import Flask, render_template, request, jsonify
import urllib.parse
from bs4 import BeautifulSoup
import concurrent.futures
import time
import http.client
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Different user agents to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

# ---- 1. Vision Transformer Setup ----

class VisionModule:
    def __init__(self):
        # Load pre-trained Vision Transformer model
        logger.info("Initializing Vision Transformer model...")
        try:
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            logger.info("Vision Transformer model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading Vision Transformer model: {e}")
            raise
    
    def process_image(self, image_path=None, image_data=None):
        """Process an image and return detected objects/concepts"""
        logger.info(f"Processing image: {image_path if image_path else 'from data'}")
        try:
            if image_path:
                if image_path.startswith('http'):
                    # Load image from URL with proper error handling and retries
                    image_data = self._download_image_with_retry(image_path)
                    image = Image.open(BytesIO(image_data))
                else:
                    # Load image from local path
                    image = Image.open(image_path)
            elif image_data:
                # Load image from binary data
                image = Image.open(BytesIO(image_data))
            else:
                raise ValueError("Either image_path or image_data must be provided")
            
            # Convert image to RGB if it's in a different mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
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
    
    def _download_image_with_retry(self, url, max_retries=3):
        """Download image with retry logic and different user agents"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Use a different user agent for each retry
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                
                # Add a small delay between retries to avoid rate limiting
                if attempt > 0:
                    time.sleep(1)
                
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                return response.content
            except (requests.RequestException, http.client.HTTPException) as e:
                last_exception = e
                logger.warning(f"Attempt {attempt+1}/{max_retries} to download image failed: {e}")
                continue
        
        # All retries failed
        logger.error(f"Failed to download image after {max_retries} attempts: {last_exception}")
        raise last_exception

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
    
    def find_similar(self, embedding, limit=5):
        """Find similar images based on embedding"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(limit, 20)  # Cap at 20 for performance
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    def search_by_label(self, search_term, limit=10):
        """Search for images by label"""
        try:
            # Get all items from the collection
            collection_items = self.collection.get()
            
            results = []
            if collection_items['ids']:
                for i, item_id in enumerate(collection_items['ids']):
                    metadata = collection_items['metadatas'][i]
                    label = metadata.get('label', '').lower()
                    
                    if search_term.lower() in label:
                        results.append({
                            "id": item_id,
                            "label": metadata.get('label', 'Unknown'),
                            "source": metadata.get('source', ''),
                            "type": metadata.get('type', 'unknown'),
                            "timestamp": metadata.get('timestamp', ''),
                            "origin": "database"
                        })
            
            # Sort by timestamp (newest first) and limit results
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching vector DB by label: {e}")
            return []

# ---- 3. Ollama LLM Integration ----

class LLMInsightGenerator:
    """Generate insights about images using Ollama LLM"""
    
    def __init__(self, model_name="mistral:latest"):
        self.model_name = model_name
        try:
            # Initialize Ollama LLM
            self.llm = OllamaLLM(model=model_name)
            logger.info(f"Initialized Ollama LLM with model: {model_name}")
            
            # Define the prompt template for image insights
            self.insight_template = PromptTemplate(
                input_variables=["label", "confidence"],
                template="""
                Generate detailed insights about an image that shows {label} (detected with {confidence}% confidence).
                
                Please provide:
                1. A short description of what this is
                2. 3-4 interesting facts about it
                3. 5 related concepts or items that are connected to this
                4. A list of 3-4 categories or fields this belongs to
                
                Format your response as a JSON with these keys:
                "type": (main category),
                "description": (brief description),
                "facts": [list of facts],
                "relations": [list of objects where each has "relationship", "related_concept", and "related_labels"]
                """
            )
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM: {e}")
            self.llm = None
    
    def generate_insights(self, label, confidence):
        """Generate insights about the detected object/concept"""
        if not self.llm:
            logger.warning("LLM not initialized, returning empty insights")
            return {}
        
        try:
            # Format the confidence score
            confidence_score = round(confidence * 100, 2)
            
            # Generate insights using the LLM
            prompt = self.insight_template.format(label=label, confidence=confidence_score)
            start_time = time.time()
            logger.info(f"Generating insights for: {label}")
            
            llm_response = self.llm.invoke(prompt)
            
            end_time = time.time()
            logger.info(f"LLM response generated in {end_time - start_time:.2f} seconds")
            
            # Extract and parse JSON from the response
            try:
                # Find JSON content within the response
                json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    json_content = llm_response
                
                # Clean up and parse the JSON
                json_content = json_content.strip().replace('```', '')
                insights = json.loads(json_content)
                return insights
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {llm_response}")
                # Attempt to create a structured response from unstructured text
                return self._create_fallback_insights(label, llm_response)
        except Exception as e:
            logger.error(f"Error generating insights with LLM: {e}")
            return self._create_basic_insights(label)
    
    def _create_fallback_insights(self, label, text):
        """Create structured insights from unstructured text"""
        lines = text.strip().split('\n')
        description = ""
        facts = []
        relations = []
        types = []
        
        # Simple parsing logic
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "description" in line.lower() or line.endswith(':'):
                current_section = "description"
                continue
            elif "fact" in line.lower() or line.startswith('1.'):
                current_section = "facts"
                continue
            elif "related" in line.lower() or "concept" in line.lower():
                current_section = "relations"
                continue
            elif "categor" in line.lower() or "field" in line.lower():
                current_section = "types"
                continue
            
            if current_section == "description":
                description += line + " "
            elif current_section == "facts" and (line.startswith('-') or re.match(r'^\d+\.', line)):
                facts.append(line.lstrip('- 1234567890.'))
            elif current_section == "relations" and (line.startswith('-') or re.match(r'^\d+\.', line)):
                related_concept = line.lstrip('- 1234567890.')
                relations.append({
                    "relationship": "RELATED_TO",
                    "related_concept": related_concept,
                    "related_labels": ["Concept"]
                })
            elif current_section == "types" and (line.startswith('-') or re.match(r'^\d+\.', line)):
                types.append(line.lstrip('- 1234567890.'))
        
        return {
            "type": types[0] if types else label.capitalize(),
            "description": description.strip(),
            "facts": facts,
            "relations": relations
        }
    
    def _create_basic_insights(self, label):
        """Create basic insights when LLM fails"""
        return {
            "type": label.capitalize(),
            "description": f"A {label} detected in the image.",
            "facts": [
                f"{label.capitalize()} is a common object/concept.",
                f"The image shows characteristics of a {label}."
            ],
            "relations": [
                {"relationship": "CATEGORY", "related_concept": label.capitalize(), "related_labels": ["Category"]}
            ]
        }

# ---- 4. Web Scraper for Similar Images ----

class ImageScraper:
    """Scrapes similar images from the web based on keywords or an image"""
    
    def __init__(self):
        self.search_engines = [
            "https://www.bing.com/images/search?q={query}&form=HDRSC2&first=1",
            "https://duckduckgo.com/?q={query}&t=h_&iax=images&ia=images"
        ]
    
    def scrape_images_by_keyword(self, keyword, limit=5, safe_search=True):
        """Scrape images based on a keyword search"""
        try:
            logger.info(f"Scraping images for keyword: {keyword}, limit: {limit}")
            
            # Format the search query
            search_query = urllib.parse.quote(f"{keyword}")
            
            # Choose a random search engine
            search_url = random.choice(self.search_engines).format(query=search_query)
            
            # Add safe search parameter if needed
            if safe_search and "bing.com" in search_url:
                search_url += "&safeSearch=Moderate"
            
            # Use a random user agent
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs based on the search engine
            images = []
            
            if "bing.com" in search_url:
                # Bing image search selectors
                image_elements = soup.select('.mimg')
                
                for element in image_elements:
                    if len(images) >= limit:
                        break
                    
                    img_src = element.get('src') or element.get('data-src')
                    if img_src and img_src.startswith(('http', 'https')):
                        # Skip SVG images which often lead to errors
                        if img_src.lower().endswith('.svg'):
                            continue
                            
                        images.append({
                            'source': img_src,
                            'label': keyword
                        })
                
                # Try alternative selectors if we didn't get enough images
                if len(images) < limit:
                    img_tags = soup.select('img.mimg')
                    for img in img_tags:
                        if len(images) >= limit:
                            break
                            
                        img_src = img.get('src') or img.get('data-src')
                        if img_src and img_src.startswith(('http', 'https')) and img_src not in [img['source'] for img in images]:
                            if img_src.lower().endswith('.svg'):
                                continue
                            
                            images.append({
                                'source': img_src,
                                'label': keyword
                            })
            
            elif "duckduckgo.com" in search_url:
                # DuckDuckGo image search selectors
                # DuckDuckGo has a different structure, we need to extract from the scripts
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and "vqd=" in script.string:
                        # Extract the vqd parameter for API requests
                        vqd_match = re.search(r'vqd="([^"]+)"', script.string)
                        if vqd_match:
                            vqd = vqd_match.group(1)
                            
                            # Make API request to get images
                            api_url = f"https://duckduckgo.com/i.js?q={search_query}&vqd={vqd}&o=json"
                            api_response = requests.get(api_url, headers={'User-Agent': random.choice(USER_AGENTS)})
                            
                            if api_response.status_code == 200:
                                try:
                                    data = api_response.json()
                                    for result in data.get('results', []):
                                        if len(images) >= limit:
                                            break
                                        
                                        img_src = result.get('image')
                                        if img_src and img_src.startswith(('http', 'https')):
                                            if img_src.lower().endswith('.svg'):
                                                continue
                                                
                                            images.append({
                                                'source': img_src,
                                                'label': keyword
                                            })
                                except:
                                    logger.warning("Failed to parse DuckDuckGo API response")
                            break
            
            # Fallback: generic image selector if we still don't have enough images
            if len(images) < limit:
                all_imgs = soup.select('img[src^="http"]')
                for img in all_imgs:
                    if len(images) >= limit:
                        break
                        
                    img_src = img.get('src')
                    if img_src and img_src.startswith(('http', 'https')) and img_src not in [img['source'] for img in images]:
                        if img_src.lower().endswith('.svg'):
                            continue
                            
                        images.append({
                            'source': img_src,
                            'label': keyword
                        })
            
            logger.info(f"Found {len(images)} images for keyword: {keyword}")
            
            # If we still don't have enough images, generate some placeholder images
            while len(images) < limit:
                # Generate a placeholder with unique ID
                placeholder_id = str(uuid.uuid4())
                images.append({
                    'source': f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(keyword)}_{placeholder_id[:8]}",
                    'label': f"{keyword} (Placeholder)",
                    'is_placeholder': True
                })
            
            return images
            
        except Exception as e:
            logger.error(f"Error scraping images for keyword {keyword}: {e}")
            # Return placeholders on failure
            return [
                {
                    'source': f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(f'{keyword}_{i}')}",
                    'label': f"{keyword} (Placeholder)",
                    'is_placeholder': True
                } for i in range(limit)
            ]
    
    def scrape_similar_images(self, image_url=None, label=None, limit=5):
        """Find similar images based on an existing image or a label"""
        # If we have a label, use it for keyword search
        if label:
            return self.scrape_images_by_keyword(label, limit)
        elif image_url:
            # For image-based search, we'll use the image URL to extract a label first
            try:
                vision_results = vision_module.process_image(image_path=image_url)
                label = vision_results['label']
                return self.scrape_images_by_keyword(label, limit)
            except Exception as e:
                logger.error(f"Error processing image for similar search: {e}")
                return self.scrape_images_by_keyword("similar image", limit)
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
            # Check if it's already a placeholder image
            if "via.placeholder.com" in image_url:
                # Just create a local copy of the placeholder
                filename = f"placeholder_{uuid.uuid4()}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Download the placeholder image
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                response = requests.get(image_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return f"/static/uploads/{filename}"
            
            # Use a random user agent
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            
            # Try to download the image
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check content type to make sure it's an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                raise ValueError(f"URL does not point to an image: {content_type}")
            
            # Create a unique filename
            file_extension = self._get_file_extension(content_type)
            filename = f"scraped_{uuid.uuid4()}{file_extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify the saved file is a valid image
            try:
                img = Image.open(filepath)
                img.verify()  # Verify it's a valid image
                img.close()
                
                # Convert non-standard formats to JPEG for better compatibility
                if file_extension not in ['.jpg', '.jpeg', '.png', '.gif']:
                    img = Image.open(filepath)
                    img = img.convert('RGB')
                    converted_filename = f"scraped_{uuid.uuid4()}.jpg"
                    converted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
                    img.save(converted_filepath, 'JPEG')
                    
                    # Remove the original file
                    os.remove(filepath)
                    return f"/static/uploads/{converted_filename}"
                
                return f"/static/uploads/{filename}"
            except Exception as e:
                logger.error(f"Downloaded file is not a valid image: {e}")
                # If verification fails, delete the file and return None
                if os.path.exists(filepath):
                    os.remove(filepath)
                return None
                
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
        elif 'bmp' in content_type:
            return '.bmp'
        elif 'tiff' in content_type:
            return '.tiff'
        else:
            return '.jpg'  # Default to jpg

# Initialize modules globally for the Flask app
try:
    vision_module = VisionModule()
    vector_db = VectorDB()
    llm_insights = LLMInsightGenerator()
    image_scraper = ImageScraper()
except Exception as e:
    logger.error(f"Error initializing modules: {e}")
    raise

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
        # Get the number of similar images to return (default is 5)
        num_similar_images = int(request.form.get('numImages', 5))
        # Cap the number to avoid overloading
        num_similar_images = min(max(num_similar_images, 1), 20)
        
        # Get the model to use for insights generation
        model = request.form.get('model', 'mistral:latest')
        
        # Get safe search setting (default is True)
        safe_search = request.form.get('safeSearch', 'true').lower() == 'true'
        
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
                    return jsonify({
                        "success": False,
                        "error": "Invalid base64 image data"
                    }), 400
                
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
                    if image_url.startswith(('http', 'https')):
                        image_data = vision_module._download_image_with_retry(image_url)
                        
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
                            f.write(image_data)
                        
                        # Process the image from the saved file
                        vision_results = vision_module.process_image(image_path=filepath)
                        source_path = f"/static/uploads/{filename}"
                    else:
                        # For local paths, process directly
                        vision_results = vision_module.process_image(image_path=image_url)
                        source_path = image_url
                except requests.exceptions.RequestException as e:
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
                "type": "url"
            }
            vector_db.add_image_data(image_id, vision_results['image_embedding'], metadata)
            
        elif 'imageFile' in request.files:
            # Image file uploaded
            file = request.files['imageFile']
            
            # Check if the file is valid
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Generate a unique filename
            filename = f"{uuid.uuid4()}_{sanitize_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Generate a unique ID for this image
            image_id = f"img_{uuid.uuid4()}"
            
            # Process the image
            try:
                vision_results = vision_module.process_image(image_path=filepath)
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Failed to process the uploaded image: {str(e)}"
                }), 400
            
            # Save results to vector database
            metadata = {
                "label": vision_results['label'],
                "source": f"/static/uploads/{filename}",
                "timestamp": datetime.now().isoformat(),
                "type": "upload"
            }
            vector_db.add_image_data(image_id, vision_results['image_embedding'], metadata)
            
        else:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Generate LLM insights instead of using hardcoded knowledge
        insights = llm_insights.generate_insights(vision_results['label'], vision_results['confidence'])
        
        # Find similar images from the vector database
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
                        "origin": "database"
                    })
        
        # Find similar images from the web using the image scraper
        # Calculate how many web images we need to reach the total requested
        additional_images_needed = max(0, num_similar_images - len(similar_images))
        
        web_images = []
        if additional_images_needed > 0:
            try:
                # Use the label for searching similar images
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
                    local_path = image_scraper.download_image(img['source'])
                    if local_path:
                        img['source'] = local_path
                    else:
                        # If download fails, use a placeholder
                        img['source'] = f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(vision_results['label'])}"
                        img['is_placeholder'] = True
            except Exception as e:
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
        
        # Return the results
        return jsonify({
            "success": True,
            "image": {
                "id": image_id,
                "label": vision_results['label'],
                "confidence": round(vision_results['confidence'] * 100, 2),
                "source": metadata['source']
            },
            "knowledge": insights,
            "similar_images": all_similar_images
        })
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

@app.route('/search', methods=['POST'])
def search_similar():
    """API endpoint to search for similar images by label"""
    try:
        search_term = request.json.get('term', '').lower()
        
        if not search_term:
            return jsonify({"success": False, "error": "No search term provided"}), 400
        
        # Use the more efficient search method from VectorDB
        results = vector_db.search_by_label(search_term)
        
        # Get the number of additional web results to fetch
        num_web_results = request.json.get('numWebResults', 5)
        num_web_results = min(max(num_web_results, 0), 10)  # Cap between 0 and 10
        
        # Add web search results if requested
        web_results = []
        if num_web_results > 0:
            try:
                web_images = image_scraper.scrape_images_by_keyword(search_term, limit=num_web_results)
                
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
                    elif img.get('is_placeholder'):
                        # Keep placeholders as they are
                        web_results.append(img)
            except Exception as e:
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
        
        return jsonify({
            "success": True,
            "results": all_results
        })
        
    except Exception as e:
        logger.error(f"Error in search_similar: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/scrape-similar', methods=['POST'])
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
                logger.warning(f"Error processing scraped image: {e}")
        
        # If we couldn't process any images, add placeholders
        if not processed_images:
            keyword = label or "image"
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
        
        return jsonify({
            "success": True,
            "images": processed_images
        })
        
    except Exception as e:
        logger.error(f"Error in scrape_similar_images: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")