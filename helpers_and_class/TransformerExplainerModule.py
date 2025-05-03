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
    