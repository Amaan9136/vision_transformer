#!/usr/bin/env python3
"""
Simple Demo Script for Vision Knowledge Explorer with Ollama

This script demonstrates a simplified version of the Vision Knowledge Explorer
that doesn't require Neo4j setup, making it easier to get started quickly.
"""

import os
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import ViTForImageClassification, ViTImageProcessor
import chromadb
from langchain.llms import Ollama

# ---- 1. Vision Transformer Setup ----

class SimpleVisionModule:
    def __init__(self):
        # Load pre-trained Vision Transformer model
        print("Loading Vision Transformer model...")
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        print("Vision Transformer model loaded successfully!")
    
    def process_image(self, image_path):
        """Process an image and return detected objects/concepts"""
        print(f"Processing image: {image_path}")
        if image_path.startswith('http'):
            # Load image from URL
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from local path
            image = Image.open(image_path)
        
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
        
        # Return the detected object/concept
        return {
            "label": label,
            "confidence": logits.softmax(dim=-1)[0][predicted_class_id].item(),
            "image_embedding": outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        }

# ---- 2. Vector Database Setup ----

class SimpleVectorDB:
    def __init__(self):
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.client = chromadb.Client()
        
        # Create a collection for image embeddings
        self.collection = self.client.create_collection(name="image_embeddings")
        print("ChromaDB initialized!")
    
    def add_image_data(self, image_id, embedding, metadata=None):
        """Add image embedding to vector database"""
        self.collection.add(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
    
    def find_similar(self, embedding, limit=5):
        """Find similar images based on embedding"""
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit
        )
        return results

# ---- 3. Simple Knowledge Base ----

class SimpleKnowledgeBase:
    """A simplified knowledge base that doesn't require Neo4j"""
    
    def __init__(self):
        # Initialize with a dictionary of concepts and their properties
        self.knowledge = {
            "car": {
                "type": "vehicle",
                "powered_by": "engine",
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
                "relations": [
                    {"relationship": "IS_A", "related_concept": "car", "related_labels": ["Category"]},
                    {"relationship": "USES", "related_concept": "electricity", "related_labels": ["Energy"]},
                    {"relationship": "MANUFACTURED_BY", "related_concept": "Tesla, Inc.", "related_labels": ["Company"]}
                ]
            },
            "smartphone": {
                "type": "electronic device",
                "purpose": "communication",
                "relations": [
                    {"relationship": "HAS_COMPONENT", "related_concept": "screen", "related_labels": ["Component"]},
                    {"relationship": "HAS_COMPONENT", "related_concept": "camera", "related_labels": ["Component"]},
                    {"relationship": "HAS_FUNCTION", "related_concept": "communication", "related_labels": ["Function"]}
                ]
            },
            "laptop": {
                "type": "electronic device",
                "purpose": "computing",
                "relations": [
                    {"relationship": "HAS_COMPONENT", "related_concept": "keyboard", "related_labels": ["Component"]},
                    {"relationship": "HAS_COMPONENT", "related_concept": "screen", "related_labels": ["Component"]},
                    {"relationship": "HAS_FUNCTION", "related_concept": "computing", "related_labels": ["Function"]}
                ]
            },
            "tree": {
                "type": "plant",
                "category": "nature",
                "relations": [
                    {"relationship": "HAS_PART", "related_concept": "leaf", "related_labels": ["Component"]},
                    {"relationship": "HAS_PART", "related_concept": "trunk", "related_labels": ["Component"]},
                    {"relationship": "IS_A", "related_concept": "plant", "related_labels": ["Category"]}
                ]
            },
            "dog": {
                "type": "animal",
                "category": "pet",
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
            return self.knowledge[concept].get("relations", [])
        
        # Check for partial matches
        for key in self.knowledge:
            if key in concept or concept in key:
                return self.knowledge[key].get("relations", [])
        
        # No match found
        return []

# ---- 4. Ollama Integration ----

class OllamaHelper:
    def __init__(self, model_name="mistral:latest"):
        print(f"Initializing Ollama with model {model_name}...")
        self.llm = Ollama(model=model_name, base_url="http://localhost:11434")
        print("Ollama initialized!")
    
    def generate_insight(self, vision_results, knowledge_results):
        """Generate insights using Ollama with Llama 3.2"""
        print("Generating insights with Ollama...")
        
        # Prepare input for the LLM
        prompt = f"""
        Analyze the following image detection and knowledge graph information, 
        and provide a brief insight about the detected object.
        
        Image Detection:
        - Detected: {vision_results['label']}
        - Confidence: {vision_results['confidence']:.2f}
        
        Knowledge Information:
        """
        
        if knowledge_results:
            for item in knowledge_results:
                prompt += f"- {item['relationship']} -> {item['related_concept']} ({', '.join(item['related_labels'])})\n"
        else:
            prompt += "No specific knowledge information available for this concept.\n"
        
        prompt += "\nPlease provide a brief insight (2-3 sentences) about what was detected in the image, incorporating the knowledge information if relevant."
        
        # Get response from Ollama
        response = self.llm.invoke(prompt)
        
        print("Insight generated!")
        return response

# ---- 5. Main Application ----

class SimpleVisionKnowledgeExplorer:
    def __init__(self, model_name="mistral:latest"):
        print("Initializing Simple Vision Knowledge Explorer...")
        self.vision_module = SimpleVisionModule()
        self.vector_db = SimpleVectorDB()
        self.knowledge_base = SimpleKnowledgeBase()
        self.ollama_helper = OllamaHelper(model_name)
        print("Simple Vision Knowledge Explorer initialized successfully!")
    
    def analyze_image(self, image_path):
        """Main function to analyze an image and provide insights"""
        print("\n==== Starting Image Analysis ====")
        
        # 1. Process the image with Vision Transformer
        vision_results = self.vision_module.process_image(image_path)
        print(f"Detected: {vision_results['label']} with confidence {vision_results['confidence']:.2f}")
        
        # 2. Store in vector database for future similarity search
        image_id = f"img_{hash(image_path)}"
        self.vector_db.add_image_data(
            image_id=image_id,
            embedding=vision_results['image_embedding'],
            metadata={"label": vision_results['label'], "source": image_path}
        )
        print("Image embedding stored in vector database.")
        
        # 3. Query knowledge base for context about detected object
        knowledge_results = self.knowledge_base.query_knowledge(vision_results['label'])
        print(f"Found {len(knowledge_results)} knowledge relationships for '{vision_results['label']}'")
        
        # 4. Generate insights using Ollama with Llama 3.2
        insight = self.ollama_helper.generate_insight(vision_results, knowledge_results)
        
        # 5. Find similar images previously processed
        similar_images = self.vector_db.find_similar(vision_results['image_embedding'])
        print(f"Found {len(similar_images['ids'][0]) if similar_images['ids'] else 0} similar images.")
        
        # 6. Return comprehensive results
        return {
            "vision_analysis": vision_results,
            "knowledge_context": knowledge_results,
            "insight": insight,
            "similar_images": similar_images
        }

# ---- Example usage ----

def main():
    # Initialize the explorer with the specified model
    print("What Ollama model would you like to use? (Default: llama3.2)")
    print("Options: llama3.2, llama3.2:8b, llama3.2:11b, phi3:mini, etc.")
    model_name = input("Model name: ").strip() or "mistral:latest"
    
    explorer = SimpleVisionKnowledgeExplorer(model_name)
    
    # Ask for image URL or path
    print("\nEnter an image URL or local file path (or press Enter for default example):")
    image_input = input().strip()
    
    if not image_input:
        # Use default example
        image_input = "https://images.unsplash.com/photo-1619767886558-efdc259cde1a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80"
        print(f"Using default example image: {image_input}")
    
    # Analyze the image
    results = explorer.analyze_image(image_input)
    
    # Print the results
    print("\n==== RESULTS ====")
    
    print("\n--- Vision Analysis ---")
    print(f"Detected: {results['vision_analysis']['label']}")
    print(f"Confidence: {results['vision_analysis']['confidence']:.2f}")
    
    print("\n--- Knowledge Context ---")
    if results['knowledge_context']:
        for item in results['knowledge_context']:
            print(f"Relationship: {item['relationship']}")
            print(f"Related concept: {item['related_concept']}")
            print(f"Related labels: {item['related_labels']}")
            print("---")
    else:
        print("No knowledge context found for this concept")
    
    print("\n--- Ollama Insight ---")
    print(results['insight'])
    
    print("\n--- Similar Images ---")
    if results['similar_images']['ids'] and results['similar_images']['ids'][0]:
        for i, image_id in enumerate(results['similar_images']['ids'][0]):
            metadata = results['similar_images']['metadatas'][0][i]
            print(f"Image ID: {image_id}")
            print(f"Label: {metadata.get('label', 'Unknown')}")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print("---")
    else:
        print("No similar images found")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
        # Provide troubleshooting guidance
        if "Connection refused" in str(e) and "11434" in str(e):
            print("\nTROUBLESHOOTING:")
            print("It looks like Ollama is not running. Please start Ollama with:")
            print("  ollama serve")
            print("In a separate terminal window.")
        elif "No such model" in str(e):
            print("\nTROUBLESHOOTING:")
            print("The specified model is not available. Please pull it first with:")
            print("  ollama pull MODEL_NAME")
            print("Where MODEL_NAME is the name of the model you want to use.")