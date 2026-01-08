# matcher.py - Face matching and verification
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import pickle
import os

class FaceMatcher:
    def __init__(self, threshold: float = 0.6):
        """
        Initialize face matcher.
        
        Args:
            threshold: Cosine similarity threshold for face verification
        """
        self.threshold = threshold
        self.embeddings_db = {}  # user_id -> list of embeddings
        self.user_info = {}      # user_id -> user information
        
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1) as float
        """
        # Flatten arrays to ensure 1D
        emb1 = np.asarray(embedding1, dtype=np.float32).flatten()
        emb2 = np.asarray(embedding2, dtype=np.float32).flatten()
        
        # Normalize
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        emb1 = emb1 / norm1
        emb2 = emb2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Clip to [0, 1] and convert to Python float
        return float(np.clip(similarity, 0.0, 1.0))
    
    def verify_face(self, probe_embedding: np.ndarray, 
                    reference_embeddings: Union[List[np.ndarray], np.ndarray]) -> Tuple[bool, float]:
        """
        Verify if probe face matches any of the reference embeddings.
        
        Args:
            probe_embedding: Embedding to verify
            reference_embeddings: List of reference embeddings or numpy array
            
        Returns:
            Tuple of (is_match, max_similarity_score)
        """
        if not reference_embeddings:
            return False, 0.0
        
        max_similarity = 0.0
        
        # Ensure we have a list
        if isinstance(reference_embeddings, np.ndarray):
            if reference_embeddings.ndim == 1:
                embeddings_list = [reference_embeddings]
            else:
                embeddings_list = list(reference_embeddings)
        else:
            embeddings_list = reference_embeddings
        
        for ref_embedding in embeddings_list:
            similarity = self.calculate_similarity(probe_embedding, ref_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
        
        is_match = max_similarity >= self.threshold
        return is_match, float(max_similarity)
    
    def identify_face(self, probe_embedding: np.ndarray, 
                      top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Identify the best matching user for a face embedding.
        
        Args:
            probe_embedding: Embedding to identify
            top_k: Number of top matches to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        matches = []
        
        for user_id, embeddings in self.embeddings_db.items():
            # Calculate similarity with each stored embedding
            max_similarity = 0.0
            
            # Convert embeddings to list if needed
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    embeddings_list = [embeddings]
                else:
                    embeddings_list = [emb for emb in embeddings]
            else:
                embeddings_list = list(embeddings)
            
            for stored_embedding in embeddings_list:
                similarity = self.calculate_similarity(probe_embedding, stored_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
            
            # Only add if above threshold
            if max_similarity >= self.threshold:
                matches.append((user_id, max_similarity))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k matches
        return matches[:top_k]
    
    def register_user(self, user_id: str, embeddings: Union[List[np.ndarray], np.ndarray], 
                      user_info: Optional[Dict] = None):
        """
        Register a new user with their embeddings.
        
        Args:
            user_id: Unique user identifier
            embeddings: List of face embeddings or numpy array
            user_info: Additional user information
        """
        # Convert to list of numpy arrays
        embeddings_list = []
        
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                # Single embedding
                embeddings_list.append(embeddings.flatten().astype(np.float32))
            else:
                # Multiple embeddings
                for i in range(embeddings.shape[0]):
                    embeddings_list.append(embeddings[i].flatten().astype(np.float32))
        else:
            # Assume it's a list
            for emb in embeddings:
                emb_array = np.asarray(emb, dtype=np.float32).flatten()
                embeddings_list.append(emb_array)
        
        # Store as numpy array for consistency
        if embeddings_list:
            embeddings_array = np.stack(embeddings_list)
        else:
            embeddings_array = np.array([])
        
        # Store in database
        self.embeddings_db[user_id] = embeddings_array
        
        # Store user information
        if user_info is None:
            user_info = {}
        self.user_info[user_id] = user_info
    
    def load_database(self, db_path: str = "embeddings"):
        """
        Load embeddings database from disk.
        
        Args:
            db_path: Path to embeddings directory
        """
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            return
        
        # Clear existing database
        self.embeddings_db.clear()
        self.user_info.clear()
        
        # Load all pickle files
        for filename in os.listdir(db_path):
            if filename.endswith('.pkl'):
                filepath = os.path.join(db_path, filename)
                try:
                    with open(filepath, 'rb') as f:
                        user_data = pickle.load(f)
                    
                    user_id = user_data.get('user_id', filename.replace('.pkl', ''))
                    embeddings = user_data.get('embeddings', np.array([]))
                    
                    # Convert to numpy array if needed
                    if isinstance(embeddings, list):
                        if embeddings:
                            embeddings = np.stack([np.asarray(emb, dtype=np.float32) for emb in embeddings])
                        else:
                            embeddings = np.array([])
                    
                    self.embeddings_db[user_id] = embeddings
                    
                    # Extract user info
                    user_info = {}
                    for key, value in user_data.items():
                        if key not in ['user_id', 'embeddings', 'num_embeddings', 'embedding_dim']:
                            user_info[key] = value
                    
                    # Add default info if missing
                    if 'name' not in user_info:
                        user_info['name'] = 'Unknown'
                    if 'registration_date' not in user_info:
                        user_info['registration_date'] = ''
                    if 'email' not in user_info:
                        user_info['email'] = ''
                    
                    self.user_info[user_id] = user_info
                    
                    # Get embedding count safely
                    if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                        num_embeddings = embeddings.shape[0]
                    else:
                        num_embeddings = 0
                    
                    print(f"Loaded user: {user_id} with {num_embeddings} embeddings")
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def save_database(self, db_path: str = "embeddings"):
        """
        Save embeddings database to disk.
        
        Args:
            db_path: Path to embeddings directory
        """
        os.makedirs(db_path, exist_ok=True)
        
        for user_id, embeddings in self.embeddings_db.items():
            # Get embedding info safely
            if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                if embeddings.ndim == 1:
                    num_embeddings = 1
                    embedding_dim = embeddings.shape[0]
                else:
                    num_embeddings = embeddings.shape[0]
                    embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
            else:
                num_embeddings = 0
                embedding_dim = 0
            
            user_data = {
                'user_id': user_id,
                'embeddings': embeddings,
                'num_embeddings': num_embeddings,
                'embedding_dim': embedding_dim
            }
            
            # Add user info
            if user_id in self.user_info:
                user_data.update(self.user_info[user_id])
            
            filepath = os.path.join(db_path, f"{user_id}.pkl")
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(user_data, f)
                
                print(f"Saved user: {user_id} with {num_embeddings} embeddings")
            except Exception as e:
                print(f"Error saving {user_id}: {e}")
    
    def get_user_count(self) -> int:
        """Get number of registered users"""
        return len(self.embeddings_db)
    
    def get_total_embeddings(self) -> int:
        """Get total number of embeddings"""
        total = 0
        for embeddings in self.embeddings_db.values():
            if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                if embeddings.ndim == 1:
                    total += 1
                else:
                    total += embeddings.shape[0]
            elif isinstance(embeddings, list):
                total += len(embeddings)
        return total
    
    def remove_user(self, user_id: str):
        """Remove user from database"""
        if user_id in self.embeddings_db:
            del self.embeddings_db[user_id]
        if user_id in self.user_info:
            del self.user_info[user_id]
    
    def clear_database(self):
        """Clear entire database"""
        self.embeddings_db.clear()
        self.user_info.clear()
    
    def get_user_embeddings(self, user_id: str) -> Optional[np.ndarray]:
        """Get embeddings for a specific user"""
        return self.embeddings_db.get(user_id)
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get info for a specific user"""
        return self.user_info.get(user_id)