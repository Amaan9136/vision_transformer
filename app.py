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
from PIL import Image
from io import BytesIO
from datetime import datetime
from transformers import ViTForImageClassification, ViTImageProcessor
import chromadb
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
                    response = requests.get(image_path)
                    image = Image.open(BytesIO(response.content))
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

# Initialize modules globally for the Flask app
vision_module = VisionModule()
vector_db = VectorDB()
knowledge_base = SimpleKnowledgeBase()

# ---- 4. Flask Routes ----

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """API endpoint to analyze an image"""
    try:
        if 'imageUrl' in request.form:
            # Image URL provided
            image_url = request.form['imageUrl']
            
            # Generate a unique ID for this image
            image_id = f"img_{uuid.uuid4()}"
            
            # Process the image
            vision_results = vision_module.process_image(image_path=image_url)
            
            # Save results to vector database
            metadata = {
                "label": vision_results['label'],
                "source": image_url,
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
        
        # Find similar images
        similar_results = vector_db.find_similar(vision_results['image_embedding'])
        
        # Prepare similar images data
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
                        "type": metadata.get('type', 'unknown')
                    })
        
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
            "similar_images": similar_images
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
                        "timestamp": metadata.get('timestamp', '')
                    })
        
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in search_similar: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")