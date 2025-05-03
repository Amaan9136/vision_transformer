from CONSTANTS.MODULES import (
    random, time, uuid, requests, http, os, np, plt, matplotlib,
    BytesIO, torch, imagehash, logger,
    Image, ImageDraw,
    ViTForImageClassification, ViTImageProcessor, ViTModel, AutoFeatureExtractor
)

from CONSTANTS.CONSTANTS import ( USER_AGENTS )

from app import flask_app

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
            attention_dir = os.path.join(flask_app.config['ATTENTION_MAPS_FOLDER'], attention_id)
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
            feature_dir = os.path.join(flask_app.config['FEATURE_MAPS_FOLDER'], feature_id)
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
            transform_dir = os.path.join(flask_app.config['TRANSFORMATION_FOLDER'], transform_id)
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
