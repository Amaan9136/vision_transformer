from CONSTANTS.MODULES import (
    logger, re, json, time, random, requests, urllib, uuid, os,
    concurrent, BytesIO, Image, imagehash, BeautifulSoup
)

from app import flask_app

from CONSTANTS.CONSTANTS import ( USER_AGENTS )

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
                filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
                
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
            filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
            
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
                    converted_filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], converted_filename)
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
