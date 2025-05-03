from CONSTANTS.MODULES import (
    logger, chromadb
)


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
