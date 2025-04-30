"""
Vision Knowledge Explorer

This mini-project combines Vision Transformers, Vector Databases, Knowledge Graphs, and CrewAI
to create a system that can analyze images, connect them with structured knowledge,
and provide insightful descriptions.
"""

import os
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import ViTForImageClassification, ViTImageProcessor
import chromadb
from neo4j import GraphDatabase
from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama

# ---- 1. Vision Transformer Setup ----

class VisionModule:
    def __init__(self):
        # Load pre-trained Vision Transformer model
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    def process_image(self, image_path):
        """Process an image and return detected objects/concepts"""
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

class VectorDB:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.Client()
        
        # Create a collection for image embeddings
        self.collection = self.client.create_collection(name="image_embeddings")
    
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

# ---- 3. Knowledge Graph Setup ----

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def query_knowledge(self, concept):
        """Query knowledge graph for information about a concept"""
        with self.driver.session() as session:
            # Example Cypher query to find information about a concept
            result = session.run(
                """
                MATCH (c:Concept {name: $concept})-[r]-(related)
                RETURN type(r) as relationship, related.name as related_concept,
                       labels(related) as related_labels
                LIMIT 10
                """,
                concept=concept
            )
            return [dict(record) for record in result]
    
    def add_concept(self, concept, properties=None):
        """Add a concept to the knowledge graph"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Concept {name: $concept})
                SET c += $properties
                RETURN c
                """,
                concept=concept,
                properties=properties or {}
            )

# ---- 4. CrewAI Orchestration ----

# Initialize language model for CrewAI
llm = Ollama(model="mistral:latest", base_url="http://localhost:11434")

# Define CrewAI agents
vision_agent = Agent(
    role="Vision Analyzer",
    goal="Accurately identify objects and concepts in images",
    backstory="I am a specialist in computer vision, trained to recognize patterns and objects in visual data.",
    verbose=True,
    llm=llm
)

vector_search_agent = Agent(
    role="Similarity Expert",
    goal="Find similar concepts and images based on embeddings",
    backstory="I specialize in finding patterns and similarities across large datasets of images and concepts.",
    verbose=True,
    llm=llm
)

knowledge_agent = Agent(
    role="Knowledge Navigator",
    goal="Extract relevant information from the knowledge graph",
    backstory="I navigate complex knowledge structures to find contextual information and connections.",
    verbose=True,
    llm=llm
)

summary_agent = Agent(
    role="Insight Synthesizer",
    goal="Create comprehensive summaries and insights from gathered information",
    backstory="I combine information from multiple sources to create meaningful narratives and insights.",
    verbose=True,
    llm=llm
)

# Example CrewAI tasks
process_image_task = Task(
    description="Analyze the image and identify key objects and concepts",
    agent=vision_agent,
    expected_output="List of detected objects with confidence scores"
)

find_similar_task = Task(
    description="Find similar images and concepts in the database",
    agent=vector_search_agent,
    expected_output="List of similar items and their relevance scores"
)

query_knowledge_task = Task(
    description="Extract relevant knowledge about the identified concepts",
    agent=knowledge_agent,
    expected_output="Structured information about the concepts from the knowledge graph"
)

generate_summary_task = Task(
    description="Synthesize all collected information into a comprehensive insight report",
    agent=summary_agent,
    expected_output="A detailed summary connecting the image analysis with broader knowledge"
)

# Create the crew
insight_crew = Crew(
    agents=[vision_agent, vector_search_agent, knowledge_agent, summary_agent],
    tasks=[process_image_task, find_similar_task, query_knowledge_task, generate_summary_task],
    verbose=2,
    process=Process.sequential
)

# ---- 5. Main Application ----

class VisionKnowledgeExplorer:
    def __init__(self):
        self.vision_module = VisionModule()
        self.vector_db = VectorDB()
        self.knowledge_graph = KnowledgeGraph()
        # Initialize with some sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize knowledge graph with some sample data for testing"""
        # Add sample concepts to knowledge graph
        concepts = {
            "car": {"type": "vehicle", "powered_by": "engine"},
            "Tesla": {"type": "electric vehicle", "manufacturer": "Tesla, Inc.", "founder": "Elon Musk"},
            "smartphone": {"type": "electronic device", "purpose": "communication"},
            "laptop": {"type": "electronic device", "purpose": "computing"},
            "tree": {"type": "plant", "category": "nature"},
            "dog": {"type": "animal", "category": "pet"}
        }
        
        for concept, props in concepts.items():
            self.knowledge_graph.add_concept(concept, props)
            
        # Add relationships (in a real scenario, this would be more complex)
        relationships = [
            ("Tesla", "IS_A", "car"),
            ("Tesla", "USES", "electricity"),
            ("smartphone", "HAS_COMPONENT", "camera"),
            ("laptop", "HAS_COMPONENT", "keyboard")
        ]
        
        with self.knowledge_graph.driver.session() as session:
            for source, rel, target in relationships:
                session.run(
                    f"""
                    MATCH (a:Concept {{name: $source}}), (b:Concept {{name: $target}})
                    MERGE (a)-[:{rel}]->(b)
                    """,
                    source=source, target=target
                )
    
    def analyze_image(self, image_path):
        """Main function to analyze an image and provide insights"""
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
        
        # 3. Query knowledge graph for context about detected object
        knowledge_results = self.knowledge_graph.query_knowledge(vision_results['label'])
        if not knowledge_results:
            # If exact match not found, try to find closest concept
            # (In a real implementation, this would use word embeddings or taxonomies)
            similar_concepts = [
                concept for concept in ["car", "Tesla", "smartphone", "laptop", "tree", "dog"]
                if concept.lower() in vision_results['label'].lower()
            ]
            if similar_concepts:
                knowledge_results = self.knowledge_graph.query_knowledge(similar_concepts[0])
        
        # 4. Find similar images previously processed
        similar_images = self.vector_db.find_similar(vision_results['image_embedding'])
        
        # 5. Return comprehensive results
        return {
            "vision_analysis": vision_results,
            "knowledge_context": knowledge_results,
            "similar_images": similar_images
        }
    
    def close(self):
        """Clean up resources"""
        self.knowledge_graph.close()


# ---- Example usage ----

def main():
    # Initialize the explorer
    explorer = VisionKnowledgeExplorer()
    
    try:
        # Analyze an image (replace with your image URL or path)
        image_url = "https://images.unsplash.com/photo-1619767886558-efdc259cde1a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80"
        results = explorer.analyze_image(image_url)
        
        # Print the results
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
        
        print("\n--- Similar Images ---")
        if results['similar_images']['ids']:
            for i, image_id in enumerate(results['similar_images']['ids'][0]):
                metadata = results['similar_images']['metadatas'][0][i]
                print(f"Image ID: {image_id}")
                print(f"Label: {metadata.get('label', 'Unknown')}")
                print(f"Source: {metadata.get('source', 'Unknown')}")
                print("---")
        else:
            print("No similar images found")
    
    finally:
        # Clean up resources
        explorer.close()


if __name__ == "__main__":
    main()