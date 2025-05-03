from CONSTANTS.MODULES import (
    logger, re, json, time, OllamaLLM, PromptTemplate
)

from CONSTANTS.CONSTANTS import ( USER_AGENTS )

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
