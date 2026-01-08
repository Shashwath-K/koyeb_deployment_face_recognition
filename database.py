# database.py - PostgreSQL database integration
import psycopg2
import numpy as np
import json
from psycopg2.extras import RealDictCursor
import traceback

class FaceDatabase:
    def __init__(self, dbname="attendance_db", user="admin", 
                 password="npg_48RFqybgkTad", host="ep-solitary-shadow-a1pohjxz.ap-southeast-1.pg.koyeb.app", port="5432"):
        self.connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.create_tables()
    
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        with self.connection.cursor() as cursor:
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Face embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    embedding BYTEA NOT NULL,
                    image_path VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Attendance records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) REFERENCES users(user_id) ON DELETE CASCADE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'PRESENT'
                )
            """)
            
            self.connection.commit()
    
    def add_user_with_embedding(self, user_id, name, embedding, email=None, image_path=None):
        """Add a new user with their face embedding"""
        try:
            # Convert numpy array to bytes
            embedding_bytes = embedding.tobytes()
            
            with self.connection.cursor() as cursor:
                # Add user
                cursor.execute("""
                    INSERT INTO users (user_id, name, email) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE 
                    SET name = EXCLUDED.name, email = EXCLUDED.email
                """, (user_id, name, email))
                
                # Add embedding
                cursor.execute("""
                    INSERT INTO face_embeddings (user_id, embedding, image_path)
                    VALUES (%s, %s, %s)
                """, (user_id, embedding_bytes, image_path))
                
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error saving to database: {e}")
            traceback.print_exc()
            self.connection.rollback()
            return False
    
    def get_all_embeddings(self):
        """Retrieve all users with their embeddings"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT u.user_id, u.name, f.embedding, f.image_path
                    FROM users u
                    JOIN face_embeddings f ON u.user_id = f.user_id
                """)
                rows = cursor.fetchall()
                
                embeddings_data = []
                for row in rows:
                    # Convert bytes back to numpy array
                    embedding_array = np.frombuffer(row['embedding'], dtype=np.float32)
                    embeddings_data.append({
                        'user_id': row['user_id'],
                        'name': row['name'],
                        'embedding': embedding_array,
                        'image_path': row['image_path']
                    })
                return embeddings_data
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        self.connection.close()