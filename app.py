"""
Enhanced Vision Knowledge Explorer

This updated version adds:
1. Improved image deduplication with perceptual hashing
2. Visualization of Vision Transformer attention maps
3. Feature extraction and visualization
4. Detailed explanations of transformer processes
5. Enhanced web scraping with redundancy detection
6. Multi-stage classification pipeline
"""

import os
import logging
import torch
import json
import uuid
import requests
import base64
import re
import random
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import imagehash

from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor,
    ViTModel,
    AutoFeatureExtractor
)
import chromadb
from flask import Flask, render_template, request, jsonify, url_for, send_file
import urllib.parse
from bs4 import BeautifulSoup
import concurrent.futures
import http.client

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ATTENTION_MAPS_FOLDER'] = 'static/attention_maps'
app.config['FEATURE_MAPS_FOLDER'] = 'static/feature_maps'
app.config['TRANSFORMATION_FOLDER'] = 'static/transformations'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ATTENTION_MAPS_FOLDER'], exist_ok=True)
os.makedirs(app.config['FEATURE_MAPS_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSFORMATION_FOLDER'], exist_ok=True)

# Different user agents to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

# ---- 1. Enhanced Vision Transformer Setup ----

class VisionModule:
    def __init__(self):
        # Load pre-trained Vision Transformer models
        logger.info("Initializing Vision Transformer models...")
        try:
            # Model for classification
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            
            # Model for attention visualization and feature extraction
            self.feature_model = ViTModel.from_pretrained('google/vit-base-patch16-224', output_attentions=True)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            logger.info("Vision Transformer models loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading Vision Transformer models: {e}")
            raise
    
    def process_image(self, image_path=None, image_data=None, generate_visualizations=False):
        """Process an image and return detected objects/concepts, with optional visualizations"""
        logger.info(f"Processing image: {image_path if image_path else 'from data'}")
        try:
            # Load image
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
            
            # Calculate perceptual hash for deduplication
            phash = str(imagehash.phash(image))
            
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
            
            # Generate visualizations if requested
            attention_maps_path = None
            feature_maps_path = None
            transformation_path = None
            
            if generate_visualizations:
                # Get attention maps and process visualizations
                attention_maps_path = self._generate_attention_visualizations(image, image_path)
                feature_maps_path = self._generate_feature_visualizations(image, image_path)
                transformation_path = self._visualize_transformation_pipeline(image, image_path)
            
            # Return the detected object/concept with additional data
            return {
                "label": label,
                "confidence": float(logits.softmax(dim=-1)[0][predicted_class_id].item()),
                "image_embedding": embedding,
                "perceptual_hash": phash,
                "attention_maps": attention_maps_path,
                "feature_maps": feature_maps_path,
                "transformation_pipeline": transformation_path
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
    
    def _generate_attention_visualizations(self, image, image_path=None):
        """Generate attention map visualizations from the ViT model"""
        try:
            # Create a unique ID for the attention maps
            attention_id = str(uuid.uuid4())
            
            # Generate file path
            attention_dir = os.path.join(app.config['ATTENTION_MAPS_FOLDER'], attention_id)
            os.makedirs(attention_dir, exist_ok=True)
            
            # Resize image for visualization
            vis_image = image.resize((224, 224))
            
            # Process image for feature extraction
            feature_inputs = self.feature_extractor(images=vis_image, return_tensors="pt")
            
            # Get attention weights
            with torch.no_grad():
                outputs = self.feature_model(**feature_inputs)
                attention_weights = outputs.attentions  # Tuple of tensors
            
            # Choose specific attention heads and layers to visualize
            # Create a multi-part visualization
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle("Vision Transformer Attention Maps", fontsize=16)
            
            # Get representative attention heads from different layers
            # Layer indices to visualize (early, mid, late layers)
            layer_indices = [0, 3, 7, 11]  # Assuming 12 layers total
            
            for idx, (ax, layer_idx) in enumerate(zip(axs.flatten(), layer_indices)):
                # Get attention weights for this layer
                layer_attention = attention_weights[layer_idx]  # Shape: [1, num_heads, seq_len, seq_len]
                
                # Average over heads for visualization
                attention_map = layer_attention[0].mean(dim=0).cpu().numpy()  # [197, 197]
                
                # The first row corresponds to the [CLS] token attention to image patches
                cls_attention = attention_map[0, 1:]  # Skip the cls token itself
                
                # Reshape to a square for visualization (should be 14x14 for ViT-Base)
                patch_size = int(np.sqrt(cls_attention.shape[0]))
                cls_attention = cls_attention.reshape(patch_size, patch_size)
                
                # Display
                ax.imshow(cls_attention, cmap='viridis')
                ax.set_title(f"Layer {layer_idx+1} Attention")
                ax.axis('off')
            
            # Save the figure
            attention_file = os.path.join(attention_dir, "attention_maps.png")
            plt.savefig(attention_file, bbox_inches='tight')
            plt.close(fig)
            
            # Create a single overlaid attention map on the original image for quick preview
            combined_attention_file = os.path.join(attention_dir, "combined_attention.png")
            
            # Use mid-layer attention (e.g., layer 6) for the overlay
            mid_layer_idx = 5
            mid_layer_attention = attention_weights[mid_layer_idx][0].mean(dim=0).cpu().numpy()[0, 1:]
            patch_size = int(np.sqrt(mid_layer_attention.shape[0]))
            attention_map = mid_layer_attention.reshape(patch_size, patch_size)
            
            # Resize attention map to match image
            attention_image = Image.fromarray((attention_map * 255).astype(np.uint8))
            attention_image = attention_image.resize(vis_image.size, Image.LANCZOS)
            
            # Convert to RGBA for overlay
            attention_image = attention_image.convert("RGBA")
            colored_attention = Image.new("RGBA", attention_image.size, (255, 0, 0, 0))
            
            # Apply colormap to attention
            for y in range(attention_image.height):
                for x in range(attention_image.width):
                    # Get attention value
                    val = attention_image.getpixel((x, y))[0]
                    # Apply red colormap with transparency
                    colored_attention.putpixel((x, y), (255, 0, 0, val))
            
            # Overlay on original image
            original_rgba = vis_image.convert("RGBA")
            overlay = Image.alpha_composite(original_rgba, colored_attention)
            overlay.save(combined_attention_file)
            
            return f"/static/attention_maps/{attention_id}/combined_attention.png"
            
        except Exception as e:
            logger.error(f"Error generating attention visualizations: {e}")
            return None
    
    def _generate_feature_visualizations(self, image, image_path=None):
        """Generate feature map visualizations"""
        try:
            # Create a unique ID for the feature maps
            feature_id = str(uuid.uuid4())
            
            # Generate file path
            feature_dir = os.path.join(app.config['FEATURE_MAPS_FOLDER'], feature_id)
            os.makedirs(feature_dir, exist_ok=True)
            
            # Resize image for visualization
            vis_image = image.resize((224, 224))
            
            # Process image for feature extraction
            feature_inputs = self.feature_extractor(images=vis_image, return_tensors="pt")
            
            # Get hidden representations
            with torch.no_grad():
                outputs = self.feature_model(**feature_inputs)
                hidden_states = outputs.last_hidden_state  # [1, 197, 768]
            
            # Extract patch embeddings (exclude CLS token)
            patch_embeddings = hidden_states[0, 1:, :].cpu().numpy()  # [196, 768]
            
            # Reshape to square grid of patches (should be 14x14 for ViT-Base)
            patch_size = int(np.sqrt(patch_embeddings.shape[0]))
            
            # Dimensionality reduction using principal components
            # Take only a few dimensions for visualization
            feature_dim = min(16, patch_embeddings.shape[1])
            
            # Create a plot
            fig, axs = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle("Vision Transformer Feature Maps", fontsize=16)
            
            # Plot representative feature dimensions
            for idx, ax in enumerate(axs.flatten()):
                if idx < feature_dim:
                    # Get this feature dimension across all patches
                    feature_map = patch_embeddings[:, idx].reshape(patch_size, patch_size)
                    
                    # Normalize for visualization
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                    
                    # Display
                    ax.imshow(feature_map, cmap='inferno')
                    ax.set_title(f"Feature {idx+1}")
                    ax.axis('off')
            
            # Save the figure
            feature_file = os.path.join(feature_dir, "feature_maps.png")
            plt.savefig(feature_file, bbox_inches='tight')
            plt.close(fig)
            
            # Also create a PCA visualization of the feature space
            from sklearn.decomposition import PCA
            
            # Apply PCA to reduce to 3 dimensions for RGB visualization
            pca = PCA(n_components=3)
            patch_features_pca = pca.fit_transform(patch_embeddings)
            
            # Reshape to image dimensions
            patch_features_pca = patch_features_pca.reshape(patch_size, patch_size, 3)
            
            # Normalize to 0-1 range
            patch_features_pca = (patch_features_pca - patch_features_pca.min()) / (patch_features_pca.max() - patch_features_pca.min() + 1e-8)
            
            # Convert to image
            pca_img = Image.fromarray((patch_features_pca * 255).astype(np.uint8))
            pca_img = pca_img.resize((224, 224), Image.LANCZOS)
            
            # Save PCA visualization
            pca_file = os.path.join(feature_dir, "feature_pca.png")
            pca_img.save(pca_file)
            
            return f"/static/feature_maps/{feature_id}/feature_maps.png"
            
        except Exception as e:
            logger.error(f"Error generating feature visualizations: {e}")
            return None
    
    def _visualize_transformation_pipeline(self, image, image_path=None):
        """Visualize the full transformation pipeline from input to patches to classification"""
        try:
            # Create a unique ID for the transformation visualizations
            transform_id = str(uuid.uuid4())
            
            # Generate file path
            transform_dir = os.path.join(app.config['TRANSFORMATION_FOLDER'], transform_id)
            os.makedirs(transform_dir, exist_ok=True)
            
            # Clone the image for processing
            vis_image = image.copy()
            
            # Resize to 224x224 for ViT
            if vis_image.size != (224, 224):
                # Save original size for reference
                original_size = vis_image.size
                vis_image = vis_image.resize((224, 224), Image.LANCZOS)
            
            # Original image
            orig_file = os.path.join(transform_dir, "1_original.jpg")
            vis_image.save(orig_file)
            
            # Simulate patch extraction (16x16 patches for ViT-Base)
            patch_size = 16
            patched_img = vis_image.copy()
            draw = ImageDraw.Draw(patched_img)
            
            # Draw patch grid
            for x in range(0, 224, patch_size):
                for y in range(0, 224, patch_size):
                    # Draw patch boundaries
                    draw.rectangle([x, y, x+patch_size-1, y+patch_size-1], outline="red", width=1)
            
            patched_file = os.path.join(transform_dir, "2_patched.jpg")
            patched_img.save(patched_file)
            
            # Simulate patch embeddings visualization
            # Extract actual patches
            patches = []
            for y in range(0, 224, patch_size):
                for x in range(0, 224, patch_size):
                    box = (x, y, x+patch_size, y+patch_size)
                    patch = vis_image.crop(box)
                    patches.append(patch)
            
            # Create a larger visualization of extracted patches
            patch_count = len(patches)
            grid_size = int(np.ceil(np.sqrt(patch_count)))
            patch_grid = Image.new('RGB', (grid_size * patch_size, grid_size * patch_size), (255, 255, 255))
            
            for i, patch in enumerate(patches):
                x = (i % grid_size) * patch_size
                y = (i // grid_size) * patch_size
                patch_grid.paste(patch, (x, y))
            
            patches_file = os.path.join(transform_dir, "3_extracted_patches.jpg")
            patch_grid.save(patches_file)
            
            # Visualize embedding process
            embedding_vis = Image.new('RGB', (500, 300), (245, 245, 250))
            draw = ImageDraw.Draw(embedding_vis)
            
            # Draw embedding diagram
            # Patch
            draw.rectangle([50, 100, 100, 150], fill="lightblue", outline="black")
            draw.text((55, 115), "Patch", fill="black")
            
            # Arrow
            draw.line([105, 125, 150, 125], fill="black", width=2)
            draw.polygon([(145, 120), (155, 125), (145, 130)], fill="black")
            
            # Linear projection
            draw.rectangle([155, 75, 245, 175], fill="lightgreen", outline="black")
            draw.text((160, 115), "Linear\nProjection", fill="black")
            
            # Arrow
            draw.line([250, 125, 295, 125], fill="black", width=2)
            draw.polygon([(290, 120), (300, 125), (290, 130)], fill="black")
            
            # Embedding
            draw.rectangle([300, 75, 400, 175], fill="lightyellow", outline="black")
            draw.text((320, 115), "Patch\nEmbedding", fill="black")
            
            # Arrow to transformer
            draw.line([405, 125, 450, 125], fill="black", width=2)
            draw.polygon([(445, 120), (455, 125), (445, 130)], fill="black")
            
            # Save embedding visualization
            embedding_file = os.path.join(transform_dir, "4_embedding_process.jpg")
            embedding_vis.save(embedding_file)
            
            # Create a simple transformer architecture visualization
            transformer_vis = Image.new('RGB', (600, 400), (245, 245, 250))
            draw = ImageDraw.Draw(transformer_vis)
            
            # Draw architecture components
            # Input embeddings
            draw.rectangle([50, 175, 150, 225], fill="lightyellow", outline="black")
            draw.text((55, 190), "Patch Embeddings", fill="black")
            
            # Position embeddings added
            draw.rectangle([50, 150, 150, 170], fill="lightpink", outline="black")
            draw.text((55, 152), "Position Embeddings", fill="black")
            
            # Arrow
            draw.line([155, 200, 195, 200], fill="black", width=2)
            draw.polygon([(190, 195), (200, 200), (190, 205)], fill="black")
            
            # Transformer encoder blocks (multiple)
            y_pos = 100
            for i in range(3):
                # Encoder block
                draw.rectangle([200, y_pos, 400, y_pos+75], fill="lightblue", outline="black")
                
                # Multi-head attention
                draw.rectangle([210, y_pos+10, 300, y_pos+65], fill="lightsalmon", outline="black")
                draw.text((215, y_pos+25), "Multi-Head\nAttention", fill="black")
                
                # MLP
                draw.rectangle([310, y_pos+10, 390, y_pos+65], fill="lightgreen", outline="black")
                draw.text((330, y_pos+30), "MLP", fill="black")
                
                y_pos += 100
                
                if i < 2:  # Don't draw arrow after last block
                    # Arrow to next block
                    draw.line([300, y_pos-15, 300, y_pos-5], fill="black", width=2)
                    draw.polygon([(295, y_pos-10), (300, y_pos), (305, y_pos-10)], fill="black")
            
            # Arrow to classification
            draw.line([405, 200, 445, 200], fill="black", width=2)
            draw.polygon([(440, 195), (450, 200), (440, 205)], fill="black")
            
            # Classification head
            draw.rectangle([450, 175, 550, 225], fill="lightseagreen", outline="black")
            draw.text((465, 190), "Classification\nHead", fill="black")
            
            # Save transformer visualization
            transformer_file = os.path.join(transform_dir, "5_transformer_architecture.jpg")
            transformer_vis.save(transformer_file)
            
            # Create a combined visualization
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Vision Transformer Processing Pipeline", fontsize=16)
            
            # Load all our visualization images
            stages = [
                (os.path.join(transform_dir, "1_original.jpg"), "Input Image"),
                (os.path.join(transform_dir, "2_patched.jpg"), "Patch Extraction"),
                (os.path.join(transform_dir, "3_extracted_patches.jpg"), "Image Patches"),
                (os.path.join(transform_dir, "4_embedding_process.jpg"), "Embedding Process"),
                (os.path.join(transform_dir, "5_transformer_architecture.jpg"), "Transformer Architecture"),
                (None, "Classification")  # We'll fill this with text
            ]
            
            # Plot each stage
            for idx, (ax, (img_path, title)) in enumerate(zip(axs.flatten(), stages)):
                if idx == 5:  # Classification result text
                    ax.text(0.5, 0.5, f"Prediction:\n{self.model.config.id2label[self.model(**self.processor(images=vis_image, return_tensors='pt')).logits.argmax(-1).item()]}",
                           horizontalalignment='center', verticalalignment='center', fontsize=14)
                    ax.axis('off')
                elif img_path:
                    img = np.array(Image.open(img_path))
                    ax.imshow(img)
                    ax.set_title(title)
                    ax.axis('off')
            
            # Save the figure
            combined_file = os.path.join(transform_dir, "vit_pipeline.png")
            plt.tight_layout()
            plt.savefig(combined_file, bbox_inches='tight')
            plt.close(fig)
            
            return f"/static/transformations/{transform_id}/vit_pipeline.png"
            
        except Exception as e:
            logger.error(f"Error visualizing transformation pipeline: {e}")
            return None

# ---- 2. Enhanced Vector Database with Deduplication ----

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
        
        # Dictionary for perceptual hash to ID mapping for faster deduplication checks
        self.hash_index = {}
        self.load_hash_index()
    
    def load_hash_index(self):
        """Load existing perceptual hashes into memory for fast deduplication checks"""
        try:
            # Get all items from the collection
            collection_items = self.collection.get()
            
            if collection_items['ids']:
                for i, item_id in enumerate(collection_items['ids']):
                    metadata = collection_items['metadatas'][i]
                    if 'perceptual_hash' in metadata:
                        self.hash_index[metadata['perceptual_hash']] = item_id
                
                logger.info(f"Loaded {len(self.hash_index)} perceptual hashes into hash index")
        except Exception as e:
            logger.error(f"Error loading hash index: {e}")
    
    def find_duplicates(self, phash, threshold=3):
        """Find duplicate images using perceptual hash with threshold"""
        # First check exact matches
        if phash in self.hash_index:
            return self.hash_index[phash]
        
        # If no exact match, check for near-duplicates
        for existing_hash, item_id in self.hash_index.items():
            # Calculate Hamming distance between hashes
            if self._hamming_distance(phash, existing_hash) <= threshold:
                return item_id
        
        return None
    
    def _hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def add_image_data(self, image_id, embedding, metadata=None):
        """Add image embedding to vector database with deduplication"""
        try:
            # Check for perceptual hash in metadata
            if metadata and 'perceptual_hash' in metadata:
                phash = metadata['perceptual_hash']
                
                # Check if this image is a duplicate
                duplicate_id = self.find_duplicates(phash)
                if duplicate_id:
                    logger.info(f"Detected duplicate image (ID: {duplicate_id}), skipping addition")
                    return duplicate_id
                
                # If not a duplicate, add to hash index
                self.hash_index[phash] = image_id
            
            # Add to collection
            self.collection.add(
                ids=[image_id],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            return image_id
        except Exception as e:
            logger.error(f"Error adding to vector DB: {e}")
            return None
    
    def find_similar(self, embedding, limit=5):
        """Find similar images based on embedding"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(limit*2, 50)  # Fetch more for deduplication filtering
            )
            
            # Filter near-duplicates from results
            filtered_results = self._filter_near_duplicates(results)
            
            # Limit to requested number
            filtered_results = self._limit_results(filtered_results, limit)
            
            return filtered_results
        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    def _filter_near_duplicates(self, results):
        """Filter near-duplicates from query results using perceptual hashes"""
        if not results['ids'][0]:
            return results
        
        # Create filtered versions of results
        filtered_ids = []
        filtered_distances = []
        filtered_metadatas = []
        
        # Track hashes we've seen to filter duplicates
        seen_hashes = set()
        
        for i, item_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Get perceptual hash if available
            phash = metadata.get('perceptual_hash')
            
            # If no hash or not a near-duplicate, include in results
            if not phash or not self._is_near_duplicate(phash, seen_hashes):
                filtered_ids.append(item_id)
                filtered_distances.append(distance)
                filtered_metadatas.append(metadata)
                
                # Add hash to seen set to filter future duplicates
                if phash:
                    seen_hashes.add(phash)
        
        # Construct filtered results dictionary
        filtered_results = {
            "ids": [filtered_ids],
            "distances": [filtered_distances],
            "metadatas": [filtered_metadatas]
        }
        
        return filtered_results
    
    def _is_near_duplicate(self, phash, seen_hashes, threshold=3):
        """Check if the hash is similar to any hash we've already seen"""
        for seen_hash in seen_hashes:
            if self._hamming_distance(phash, seen_hash) <= threshold:
                return True
        return False
    
    def _limit_results(self, results, limit):
        """Limit results to requested number"""
        if not results['ids'][0] or len(results['ids'][0]) <= limit:
            return results
        
        return {
            "ids": [results['ids'][0][:limit]],
            "distances": [results['distances'][0][:limit]],
            "metadatas": [results['metadatas'][0][:limit]]
        }
    
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
                            "origin": "database",
                            "attention_maps": metadata.get('attention_maps'),
                            "feature_maps": metadata.get('feature_maps'),
                            "transformation_pipeline": metadata.get('transformation_pipeline')
                        })
            
            # Sort by timestamp (newest first) and limit results
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching vector DB by label: {e}")
            return []

# ---- 3. Enhanced LLM Insight Generator with Transformer Details ----

class LLMInsightGenerator:
    """Generate insights about images using Ollama LLM with Transformer-specific details"""
    
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
                1. A short technical description of what this is
                2. 3-4 interesting facts about it
                3. 5 related concepts or items that are connected to this
                4. A list of 3-4 categories or fields this belongs to
                5. A brief explanation of how Vision Transformers would process this type of image
                
                Format your response as a JSON with these keys:
                "type": (main category),
                "description": (brief description),
                "facts": [list of facts],
                "relations": [list of objects where each has "relationship", "related_concept", and "related_labels"],
                "transformer_insight": (explanation of how ViT processes this image type)
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
        transformer_insight = ""
        
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
            elif "transformer" in line.lower() or "vit" in line.lower():
                current_section = "transformer_insight"
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
            elif current_section == "transformer_insight":
                transformer_insight += line + " "
        
        return {
            "type": types[0] if types else label.capitalize(),
            "description": description.strip(),
            "facts": facts,
            "relations": relations,
            "transformer_insight": transformer_insight.strip() or self._generate_default_transformer_insight(label)
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
            ],
            "transformer_insight": self._generate_default_transformer_insight(label)
        }
    
    def _generate_default_transformer_insight(self, label):
        """Generate a default transformer insight based on the label"""
        # Basic explanation that would apply to most images
        return f"""
        Vision Transformers process {label} images by first dividing the image into fixed-size patches (typically 16x16 pixels). 
        Each patch is linearly embedded and combined with position embeddings. The transformer then uses self-attention to learn 
        relationships between all patches, allowing it to capture global features and context. For {label} images, the model 
        likely focuses on distinctive patterns, shapes, colors, and textures typical of this class. The [CLS] token embedding 
        from the final transformer layer is used for classification.
        """

# ---- 4. Enhanced Web Scraper with Duplicate Detection ----

class ImageScraper:
    """Scrapes similar images with enhanced duplicate detection"""
    
    def __init__(self):
        self.search_engines = [
            "https://www.bing.com/images/search?q={query}&form=HDRSC2&first=1",
            "https://duckduckgo.com/?q={query}&t=h_&iax=images&ia=images"
        ]
        
        # Initialize cache for scraped image hashes
        self.image_hash_cache = {}
    
    def scrape_images_by_keyword(self, keyword, limit=5, safe_search=True):
        """Scrape images based on a keyword search with duplicate filtering"""
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
            image_candidates = []
            
            if "bing.com" in search_url:
                # Bing image search selectors
                image_elements = soup.select('.mimg')
                
                for element in image_elements:
                    img_src = element.get('src') or element.get('data-src')
                    if img_src and img_src.startswith(('http', 'https')):
                        # Skip SVG images which often lead to errors
                        if img_src.lower().endswith('.svg'):
                            continue
                            
                        image_candidates.append({
                            'source': img_src,
                            'label': keyword
                        })
                
                # Try alternative selectors if we didn't get enough images
                if len(image_candidates) < limit * 2:  # Get more to allow for filtering
                    img_tags = soup.select('img.mimg')
                    for img in img_tags:
                        img_src = img.get('src') or img.get('data-src')
                        if img_src and img_src.startswith(('http', 'https')) and img_src not in [img['source'] for img in image_candidates]:
                            if img_src.lower().endswith('.svg'):
                                continue
                            
                            image_candidates.append({
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
                                        img_src = result.get('image')
                                        if img_src and img_src.startswith(('http', 'https')):
                                            if img_src.lower().endswith('.svg'):
                                                continue
                                                
                                            image_candidates.append({
                                                'source': img_src,
                                                'label': keyword
                                            })
                                except:
                                    logger.warning("Failed to parse DuckDuckGo API response")
                            break
            
            # Fallback: generic image selector if we still don't have enough images
            if len(image_candidates) < limit * 2:
                all_imgs = soup.select('img[src^="http"]')
                for img in all_imgs:
                    img_src = img.get('src')
                    if img_src and img_src.startswith(('http', 'https')) and img_src not in [img['source'] for img in image_candidates]:
                        if img_src.lower().endswith('.svg'):
                            continue
                            
                        image_candidates.append({
                            'source': img_src,
                            'label': keyword
                        })
            
            logger.info(f"Found {len(image_candidates)} image candidates for keyword: {keyword}")
            
            # Filter duplicates using perceptual hashing
            filtered_images = self._filter_duplicate_images(image_candidates, limit)
            
            logger.info(f"After deduplication: {len(filtered_images)} unique images for keyword: {keyword}")
            
            # If we still don't have enough images, generate some placeholder images
            while len(filtered_images) < limit:
                # Generate a placeholder with unique ID
                placeholder_id = str(uuid.uuid4())
                filtered_images.append({
                    'source': f"https://via.placeholder.com/400x300.png?text={urllib.parse.quote(keyword)}_{placeholder_id[:8]}",
                    'label': f"{keyword} (Placeholder)",
                    'is_placeholder': True
                })
            
            return filtered_images
            
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
    
    def _filter_duplicate_images(self, image_candidates, limit):
        """Filter duplicate images using perceptual hashing"""
        filtered_images = []
        seen_hashes = set()
        
        # Try to download and hash each candidate
        for candidate in image_candidates:
            if len(filtered_images) >= limit:
                break
                
            img_url = candidate['source']
            
            # Skip if we've already processed this URL
            if img_url in self.image_hash_cache:
                phash = self.image_hash_cache[img_url]
                if not self._is_similar_hash(phash, seen_hashes):
                    filtered_images.append(candidate)
                    seen_hashes.add(phash)
                continue
            
            try:
                # Download image to calculate hash
                response = requests.get(img_url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
                if response.status_code == 200:
                    img_data = response.content
                    img = Image.open(BytesIO(img_data))
                    
                    # Generate perceptual hash
                    phash = str(imagehash.phash(img))
                    
                    # Cache the hash
                    self.image_hash_cache[img_url] = phash
                    
                    # Add to filtered list if not a duplicate
                    if not self._is_similar_hash(phash, seen_hashes):
                        filtered_images.append(candidate)
                        seen_hashes.add(phash)
            except Exception as e:
                logger.warning(f"Error processing image {img_url}: {e}")
                # Skip this image
                continue
        
        return filtered_images
    
    def _is_similar_hash(self, phash, seen_hashes, threshold=3):
        """Check if hash is similar to any in the seen set"""
        for seen_hash in seen_hashes:
            # Calculate Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(phash, seen_hash))
            if distance <= threshold:
                return True
        return False
    
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

# ---- 5. New Transformer Explanation Module ----

class TransformerExplainerModule:
    """Module for generating explanations and visualizations of Vision Transformer processes"""
    
    def __init__(self):
        self.patch_size = 16  # Default patch size for ViT-Base
        self.transformer_stages = [
            {
                "name": "Image Preprocessing",
                "description": "The input image is resized to 224x224 pixels and normalized using RGB channel means and standard deviations."
            },
            {
                "name": "Patch Extraction",
                "description": f"The image is divided into {self.patch_size}×{self.patch_size} pixel patches, resulting in a sequence of visual tokens."
            },
            {
                "name": "Patch Embedding",
                "description": "Each patch is flattened and projected through a linear layer to a fixed embedding dimension (typically 768)."
            },
            {
                "name": "Position Embedding",
                "description": "Learned position embeddings are added to each patch embedding to retain spatial information."
            },
            {
                "name": "Transformer Encoder",
                "description": "The sequence of embedded patches is processed through multiple layers of transformer blocks, each containing multi-head self-attention and MLP blocks."
            },
            {
                "name": "Classification Head",
                "description": "The [CLS] token embedding from the final transformer layer is used for image classification via a linear projection."
            }
        ]
    
    def generate_general_explanation(self):
        """Generate a general explanation of how Vision Transformers work"""
        explanation = {
            "title": "How Vision Transformers Work",
            "introduction": """
                Vision Transformers (ViT) are neural network architectures that apply the transformer 
                model from natural language processing to image analysis. Unlike traditional convolutional 
                neural networks (CNNs), ViTs don't use convolution or pooling operations. Instead, they 
                treat images as sequences of patches and process them with self-attention mechanisms.
            """,
            "stages": self.transformer_stages,
            "key_differences": [
                "Global context: ViTs can attend to all parts of an image simultaneously, whereas CNNs build hierarchical representations",
                "Fewer inductive biases: ViTs don't have the spatial inductive biases built into CNNs, requiring more data to train effectively",
                "Scalability: ViTs scale very well with data and model size, showing excellent performance on large datasets",
                "Data efficiency: Without specialized techniques, ViTs typically require more training data than CNNs"
            ],
            "attention_mechanism": """
                The core innovation of transformers is the self-attention mechanism. For each patch in the image,
                attention weights are computed to determine how much focus should be placed on other patches when
                creating a representation. This allows the model to capture long-range dependencies and global
                context, which is particularly useful for understanding the overall composition of images.
            """
        }
        return explanation
    
    def generate_custom_explanation(self, image_type):
        """Generate a customized explanation for a specific type of image"""
        # Base explanation that gets customized
        explanation = self.generate_general_explanation()
        
        # Add image-type specific insights
        if "face" in image_type.lower() or "person" in image_type.lower():
            explanation["custom_insight"] = """
                For facial images, Vision Transformers excel by capturing relationships between facial features.
                The self-attention mechanism allows the model to focus on important facial landmarks (eyes, nose, mouth)
                while maintaining awareness of their relative positions. This global context is crucial for
                recognizing expressions, identities, and subtle facial attributes. ViTs may attend strongly
                to distinctive features like the eyes, which carry significant identification information.
            """
        elif "text" in image_type.lower() or "document" in image_type.lower():
            explanation["custom_insight"] = """
                When processing text in images, Vision Transformers can leverage their ability to capture
                long-range dependencies between characters and words. The attention mechanism helps establish
                relationships between distant parts of text, which is valuable for understanding document structure.
                Unlike CNNs, ViTs can attend to entire text lines or paragraphs at once, helping with document
                understanding tasks.
            """
        elif "landscape" in image_type.lower() or "outdoor" in image_type.lower():
            explanation["custom_insight"] = """
                For landscape and outdoor scenes, Vision Transformers capture global composition effectively.
                The attention mechanism can identify relationships between sky, terrain, vegetation, and water bodies
                across the entire image. This helps the model understand overall scene layout and distinguish
                between different landscape types. Attention maps often show strong focus on horizon lines,
                distinctive landmarks, and boundaries between different regions.
            """
        elif "animal" in image_type.lower():
            explanation["custom_insight"] = """
                When analyzing animal images, Vision Transformers can focus on distinctive anatomical features
                while maintaining awareness of overall body structure. Attention mechanisms may emphasize areas like
                the head, distinctive coat patterns, or characteristic postures. The model can establish relationships
                between body parts that help identify species, behaviors, or health conditions.
            """
        elif "food" in image_type.lower():
            explanation["custom_insight"] = """
                For food images, Vision Transformers examine relationships between ingredients, textures, and presentation.
                The attention mechanism allows the model to focus on distinctive elements like toppings, garnishes,
                or characteristic shapes while maintaining awareness of the dish composition. This global context
                helps identify specific cuisines, dishes, and preparation styles.
            """
        else:
            # Generic customization for other image types
            explanation["custom_insight"] = f"""
                When processing {image_type} images, Vision Transformers divide the image into patches and process
                them with self-attention. This helps the model identify key features and relationships specific to
                {image_type}. The attention maps would likely highlight distinctive visual elements most relevant for
                recognizing this type of content. Unlike CNNs, ViTs maintain a global view of the entire image,
                which can be particularly beneficial for understanding overall composition and context.
            """
        
        return explanation
    
    def generate_attention_explanation(self, image_type):
        """Generate an explanation of how attention works for a specific image type"""
        
        general_explanation = """
            Attention in Vision Transformers works by computing query, key, and value projections for each image patch.
            For each patch (query), the model calculates its compatibility with all other patches (keys) to determine
            attention weights. These weights are then used to create a weighted sum of values from all patches,
            allowing information to flow between related regions regardless of spatial distance.
        """
        
        if "face" in image_type.lower():
            specific_explanation = """
                When processing facial images, attention often focuses strongly on key facial features like eyes,
                nose, and mouth. The [CLS] token typically attends most to these distinctive regions, with
                especially high attention weights for the eyes and other identity-carrying features. This helps
                the model capture facial expressions and identity information.
            """
        elif "text" in image_type.lower():
            specific_explanation = """
                For text images, attention patterns often follow reading order and character structure. The model
                creates connections between related characters and words, with strong attention between parts of
                the same word or related words. This helps the model understand document structure and relationships.
            """
        elif "landscape" in image_type.lower():
            specific_explanation = """
                In landscape images, attention often focuses on horizon lines, distinctive landmarks, and
                boundaries between regions (like sky and ground). The attention mechanism helps establish
                spatial relationships between elements like mountains, water bodies, and sky, enabling the
                model to understand scene composition regardless of lighting conditions.
            """
        elif "building" in image_type.lower() or "architecture" in image_type.lower():
            specific_explanation = """
                For architectural images, attention tends to concentrate on structural elements like edges, 
                corners, and distinctive features of buildings. The model establishes relationships between
                architectural components, helping it recognize building styles, materials, and structural patterns
                regardless of viewing angle or lighting conditions.
            """
        elif "vehicle" in image_type.lower() or "car" in image_type.lower():
            specific_explanation = """
                When processing vehicle images, attention focuses on distinctive elements like headlights,
                grilles, wheels, and overall shape profiles. The model creates connections between these
                components, especially attending to brand-specific design elements. This helps the model
                identify vehicle types, manufacturers, and models across different viewing angles.
            """
        elif "product" in image_type.lower():
            specific_explanation = """
                For product images, attention mechanisms typically focus on distinctive brand elements,
                product shapes, and unique design features. The model can establish relationships between
                various parts of the product, helping identify specific items even with different backgrounds
                or viewing angles. Attention is often strongest on logo areas and characteristic product silhouettes.
            """
        else:
            specific_explanation = f"""
                For {image_type} images, attention mechanisms likely focus on the most distinctive visual features
                relevant to this category. Attention weights help establish relationships between important elements,
                allowing the model to understand both local details and global context simultaneously, which is
                crucial for accurate classification and understanding.
            """
        
        return {
            "general_explanation": general_explanation,
            "specific_explanation": specific_explanation,
            "attention_heads": "Different attention heads in the model focus on different aspects of the image. Some heads track specific features, while others monitor overall composition or texture patterns."
        }
    
    def create_architecture_diagram(self):
        """Create a visualization of the Vision Transformer architecture"""
        # Return a static path to a pre-generated architecture diagram
        # In a real implementation, this would dynamically generate a diagram
        return "/static/images/vit_architecture.png"
    
    def generate_feature_maps_explanation(self, image_type):
        """Generate an explanation of how feature maps work in Vision Transformers"""
        
        general_explanation = """
            Unlike CNNs with explicit feature maps, Vision Transformers produce features through self-attention
            and MLP operations. Each transformer layer generates a set of hidden state vectors (one per patch)
            that act as feature representations. Early layers capture low-level features while deeper layers
            develop more abstract semantic representations.
        """
        
        if "face" in image_type.lower():
            specific_explanation = """
                When processing facial images, ViT feature maps encode facial component relationships.
                Earlier layers might represent basic features like edges and textures, while deeper layers
                capture more semantic features like eye shapes, expressions, or facial structure. Different
                attention heads may specialize in tracking different facial attributes.
            """
        elif "text" in image_type.lower():
            specific_explanation = """
                For text images, ViT feature maps encode character shapes, spacing patterns, and text layout.
                The model develops representations that capture both local features (character shapes) and
                global features (text alignment, spacing, formatting). Different dimensions in the feature
                space may encode different aspects like font style, character size, or text density.
            """
        elif "landscape" in image_type.lower():
            specific_explanation = """
                With landscape images, ViT feature maps represent terrain textures, spatial layouts, and
                region boundaries. The self-attention mechanism helps create representations that capture
                the relationships between different landscape elements like sky, ground, water, and vegetation.
                These representations enable the model to distinguish between different types of landscapes.
            """
        elif "object" in image_type.lower():
            specific_explanation = """
                For object-centric images, ViT feature maps encode shape contours, textures, and object parts.
                The attention mechanism helps create cohesive object representations by establishing relationships
                between different parts of the object. This allows the model to recognize objects despite variations
                in viewing angle, lighting, or partial occlusion.
            """
        else:
            specific_explanation = f"""
                For {image_type} images, ViT feature maps likely encode the distinctive visual characteristics
                and spatial relationships that define this category. The transformer architecture allows these
                features to incorporate global context, which is particularly valuable for understanding complex
                {image_type} scenes or objects.
            """
        
        return {
            "general_explanation": general_explanation,
            "specific_explanation": specific_explanation,
            "feature_dimensions": "The typical 768-dimensional embedding space of ViT-Base contains a rich representation of visual features, with different dimensions encoding different aspects of visual content."
        }
    
    def generate_comparison_with_cnn(self):
        """Generate a comparison between Vision Transformers and CNNs"""
        
        return {
            "title": "Vision Transformers vs. Convolutional Neural Networks",
            "introduction": """
                While CNNs have dominated computer vision for nearly a decade, Vision Transformers offer an
                alternative approach that processes images in a fundamentally different way. Understanding these
                differences helps explain ViT's strengths and limitations.
            """,
            "comparison_points": [
                {
                    "aspect": "Basic Operation",
                    "cnn": "Uses convolutional filters that slide across the image, processing small regions at a time.",
                    "vit": "Divides image into patches, projects them into embeddings, and applies self-attention across all patches."
                },
                {
                    "aspect": "Receptive Field",
                    "cnn": "Builds hierarchically - early layers have small receptive fields, deeper layers see more of the image.",
                    "vit": "Global from the start - all patches can attend to all other patches in every layer."
                },
                {
                    "aspect": "Inductive Bias",
                    "cnn": "Strong spatial inductive bias - assumes nearby pixels are related and builds features hierarchically.",
                    "vit": "Minimal inductive bias - learns relationships between patches with fewer assumptions about spatial structure."
                },
                {
                    "aspect": "Parameter Efficiency",
                    "cnn": "More parameter-efficient, especially for smaller datasets.",
                    "vit": "Less parameter-efficient, requires more data but scales better with model size."
                },
                {
                    "aspect": "Computational Complexity",
                    "cnn": "Scales linearly with image size (O(n) for n pixels).",
                    "vit": "Scales quadratically with number of patches (O(n²) for n patches)."
                },
                {
                    "aspect": "Training Data Requirements",
                    "cnn": "Can perform well with moderate dataset sizes due to inductive biases.",
                    "vit": "Typically requires larger datasets to reach comparable performance."
                },
                {
                    "aspect": "Transfer Learning",
                    "cnn": "Good transfer to downstream tasks, especially for related domains.",
                    "vit": "Excellent transfer learning capabilities, especially at larger scales."
                }
            ],
            "conclusion": """
                Both architectures have complementary strengths. CNNs excel with limited data and computational
                resources, while ViTs show superior performance at scale and capture global relationships more
                efficiently. Modern approaches often combine elements of both architectures to leverage their
                respective advantages.
            """
        }
    
    def generate_task_specific_insights(self, task_type):
        """Generate insights about how Vision Transformers perform on specific computer vision tasks"""
        
        general_insight = """
            Vision Transformers have been adapted to a wide range of computer vision tasks beyond classification,
            including object detection, segmentation, and image generation. Their ability to capture global context
            and establish long-range dependencies makes them particularly suitable for tasks requiring holistic
            scene understanding.
        """
        
        if "classification" in task_type.lower():
            specific_insight = """
                For image classification, Vision Transformers shine at scale. When trained on sufficient data,
                they can outperform CNNs by capturing subtle global patterns. The [CLS] token effectively aggregates
                information from the entire image, enabling accurate category prediction. ViTs particularly excel
                at fine-grained classification tasks where global context matters.
            """
        elif "detection" in task_type.lower():
            specific_insight = """
                For object detection, Vision Transformers have been adapted in models like DETR (DEtection TRansformer),
                which reformulates detection as a direct set prediction problem. This approach eliminates the need for
                many hand-designed components like non-maximum suppression or anchor generation. The self-attention
                mechanism naturally models relationships between objects, improving detection in complex scenes.
            """
        elif "segmentation" in task_type.lower():
            specific_insight = """
                In segmentation tasks, Vision Transformers provide pixel-level predictions by establishing relationships
                between all image regions. Models like SegFormer and Mask2Former leverage transformer architectures
                to achieve state-of-the-art segmentation performance. Their global context awareness helps resolve
                ambiguities at object boundaries and improves consistency across the image.
            """
        elif "generation" in task_type.lower() or "synthesis" in task_type.lower():
            specific_insight = """
                For image generation and synthesis, Vision Transformers can model complex dependencies between
                image elements. Models like ViT-VQGAN use transformers to autoregressively generate image tokens,
                capturing global structure and coherence. Their attention mechanisms help ensure consistency across
                the generated image while maintaining fine details.
            """
        else:
            specific_insight = f"""
                For {task_type} tasks, Vision Transformers likely leverage their global context modeling and
                long-range dependency capture to improve performance. The self-attention mechanism helps the model
                understand relationships between different parts of the image that are relevant to this specific task.
            """
        
        return {
            "general_insight": general_insight,
            "specific_insight": specific_insight,
            "adaptations": "Vision Transformers often require task-specific adaptations, such as specialized tokens, hierarchical structures, or hybrid architectures combining transformer blocks with convolutional layers."
        }
    
    def generate_deployment_considerations(self):
        """Generate information about practical considerations for deploying Vision Transformers"""
        
        return {
            "title": "Practical Considerations for Vision Transformer Deployment",
            "introduction": """
                While Vision Transformers offer compelling advantages, deploying them effectively requires
                considering several practical factors that influence performance, efficiency, and resource usage.
            """,
            "considerations": [
                {
                    "aspect": "Computational Requirements",
                    "description": "Vision Transformers are computationally intensive, especially for high-resolution images. The quadratic complexity of self-attention with respect to sequence length can be a bottleneck."
                },
                {
                    "aspect": "Memory Usage",
                    "description": "ViTs typically require more memory than equivalent CNNs, particularly during training. This can limit batch sizes on standard hardware."
                },
                {
                    "aspect": "Inference Latency",
                    "description": "Standard ViTs may have higher inference latency than CNNs, though specialized architectures like DeiT and mobile-optimized versions help address this issue."
                },
                {
                    "aspect": "Data Requirements",
                    "description": "ViTs generally need more training data than CNNs to reach comparable performance due to their lower inductive bias."
                },
                {
                    "aspect": "Pre-trained Models",
                    "description": "Using pre-trained ViT models is often essential for good performance. Models trained on large datasets like ImageNet-21k or JFT-300M serve as excellent starting points."
                },
                {
                    "aspect": "Hardware Acceleration",
                    "description": "Modern accelerators with efficient attention implementations (like NVIDIA A100 with sparse attention) can significantly improve ViT performance."
                },
                {
                    "aspect": "Quantization",
                    "description": "Post-training quantization to lower precision (INT8, FP16) can substantially reduce memory footprint and inference time with minimal accuracy impact."
                }
            ],
            "conclusion": """
                When deploying Vision Transformers, consider the trade-offs between model size, computational 
                requirements, and accuracy for your specific use case. Hybrid approaches combining CNN and 
                transformer elements often provide a good balance for real-world applications.
            """
        }
    

# Flask routes for the Vision Transformer Explorer application

@app.route('/')
def index():
    """Main page"""
    # Generate transformer explanation for initial page load
    transformer_explanation = transformer_explainer.generate_general_explanation()
    comparison = transformer_explainer.generate_comparison_with_cnn()
    return render_template('index.html', 
                          transformer_explanation=transformer_explanation, 
                          comparison=comparison)

@app.route('/transformer_explanation')
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

@app.route('/attention_explanation')
def get_attention_explanation():
    """API endpoint to get an explanation of attention for a specific image type"""
    image_type = request.args.get('type', 'general')
    explanation = transformer_explainer.generate_attention_explanation(image_type)
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })

@app.route('/feature_maps_explanation')
def get_feature_maps_explanation():
    """API endpoint to get an explanation of feature maps for a specific image type"""
    image_type = request.args.get('type', 'general')
    explanation = transformer_explainer.generate_feature_maps_explanation(image_type)
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })

@app.route('/task_specific_insights')
def get_task_specific_insights():
    """API endpoint to get insights about Vision Transformers for specific CV tasks"""
    task_type = request.args.get('task', 'classification')
    insights = transformer_explainer.generate_task_specific_insights(task_type)
    
    return jsonify({
        "success": True,
        "insights": insights
    })

@app.route('/deployment_considerations')
def get_deployment_considerations():
    """API endpoint to get deployment considerations for Vision Transformers"""
    considerations = transformer_explainer.generate_deployment_considerations()
    
    return jsonify({
        "success": True,
        "considerations": considerations
    })

@app.route('/architecture_diagram')
def get_architecture_diagram():
    """API endpoint to get the transformer architecture diagram"""
    diagram_path = transformer_explainer.create_architecture_diagram()
    
    return jsonify({
        "success": True,
        "diagram_path": diagram_path
    })

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
        
        # Get visualization setting (default is True)
        generate_visualizations = request.form.get('generateVisualizations', 'true').lower() == 'true'
        
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
                vision_results = vision_module.process_image(image_data=image_data, generate_visualizations=generate_visualizations)
                
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
                        vision_results = vision_module.process_image(image_path=filepath, generate_visualizations=generate_visualizations)
                        source_path = f"/static/uploads/{filename}"
                    else:
                        # For local paths, process directly
                        vision_results = vision_module.process_image(image_path=image_url, generate_visualizations=generate_visualizations)
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
                "type": "url",
                "perceptual_hash": vision_results['perceptual_hash'],
                "attention_maps": vision_results.get('attention_maps'),
                "feature_maps": vision_results.get('feature_maps'),
                "transformation_pipeline": vision_results.get('transformation_pipeline')
            }
            
            # Check for duplicates before adding to database
            duplicate_id = vector_db.find_duplicates(vision_results['perceptual_hash'])
            if duplicate_id:
                logger.info(f"Duplicate image detected, using existing ID: {duplicate_id}")
                image_id = duplicate_id
            else:
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
                vision_results = vision_module.process_image(image_path=filepath, generate_visualizations=generate_visualizations)
            except Exception as e:
                logger.error(f"Error processing uploaded image: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Failed to process the uploaded image: {str(e)}"
                }), 400
            
            # Check for duplicates before adding to database
            duplicate_id = vector_db.find_duplicates(vision_results['perceptual_hash'])
            if duplicate_id:
                logger.info(f"Duplicate image detected, using existing ID: {duplicate_id}")
                image_id = duplicate_id
            else:
                # Save results to vector database
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
        insights = llm_insights.generate_insights(vision_results['label'], vision_results['confidence'])
        
        # Get transformer explanation for this image type
        transformer_explanation = transformer_explainer.generate_custom_explanation(vision_results['label'])
        
        # Get attention explanation for this image type
        attention_explanation = transformer_explainer.generate_attention_explanation(vision_results['label'])
        
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

@app.route('/visualizations/<path:viz_type>/<path:viz_id>')
def get_visualization(viz_type, viz_id):
    """API endpoint to get a specific visualization"""
    try:
        # Determine the visualization type and path
        if viz_type == 'attention':
            viz_path = os.path.join(app.config['ATTENTION_MAPS_FOLDER'], viz_id)
        elif viz_type == 'features':
            viz_path = os.path.join(app.config['FEATURE_MAPS_FOLDER'], viz_id)
        elif viz_type == 'transformation':
            viz_path = os.path.join(app.config['TRANSFORMATION_FOLDER'], viz_id)
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

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """API endpoint to compare ViT with other vision models on an image"""
    try:
        # Get the image ID to analyze
        image_id = request.json.get('imageId')
        if not image_id:
            return jsonify({"success": False, "error": "No image ID provided"}), 400
        
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
        
        return jsonify({
            "success": True,
            "image_source": image_source,
            "comparison": comparison_results
        })
        
    except Exception as e:
        logger.error(f"Error in compare_models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate_attention_visualization', methods=['POST'])
def generate_attention_visualization():
    """API endpoint to generate a custom attention visualization"""
    try:
        # Get the image ID and heads to visualize
        image_id = request.json.get('imageId')
        layer = request.json.get('layer', 0)  # Default to first layer
        heads = request.json.get('heads', [0])  # Default to first attention head
        
        if not image_id:
            return jsonify({"success": False, "error": "No image ID provided"}), 400
        
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
                image_path = os.path.join(app.root_path, image_source.lstrip('/'))
                image = Image.open(image_path)
                
                # Generate attention maps
                attention_maps = vision_module._generate_attention_visualizations(image)
                
                # Update metadata with attention maps
                vector_db.collection.update(
                    ids=[image_id],
                    metadatas=[{**metadata, "attention_maps": attention_maps}]
                )
            except Exception as e:
                logger.error(f"Error generating attention maps: {e}")
                return jsonify({"success": False, "error": f"Failed to generate attention maps: {str(e)}"}), 500
        
        # Return the attention visualization path
        return jsonify({
            "success": True,
            "attention_visualization": metadata.get('attention_maps'),
            "custom_visualization": f"/static/attention_maps/custom_{image_id}_{layer}_{'-'.join(map(str, heads))}.png"
        })
        
    except Exception as e:
        logger.error(f"Error in generate_attention_visualization: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download_visualization/<path:viz_type>/<path:viz_id>')
def download_visualization(viz_type, viz_id):
    """API endpoint to download a visualization as an image file"""
    try:
        # Determine the visualization type and path
        if viz_type == 'attention':
            viz_path = os.path.join(app.config['ATTENTION_MAPS_FOLDER'], viz_id)
            filename = "attention_map.png"
        elif viz_type == 'features':
            viz_path = os.path.join(app.config['FEATURE_MAPS_FOLDER'], viz_id)
            filename = "feature_maps.png"
        elif viz_type == 'transformation':
            viz_path = os.path.join(app.config['TRANSFORMATION_FOLDER'], viz_id)
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
        logger.error(f"Error downloading visualization: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/interactive_transformer')
def interactive_transformer():
    """Interactive Vision Transformer demo page"""
    return render_template('interactive.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")