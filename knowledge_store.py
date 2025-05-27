import sqlite3
import numpy as np
import pickle
from typing import List, Dict, Tuple
import os

class KnowledgeStore:
    def __init__(self, db_path: str = "knowledge_base.db"):
        """Initialize the knowledge store with a SQLite database."""
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        """Create the database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create table for knowledge base entries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def store_knowledge(self, title: str, content: str, embedding: np.ndarray):
        """Store a knowledge base entry with its embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Convert numpy array to bytes for storage
            embedding_bytes = pickle.dumps(embedding)
            cursor.execute(
                'INSERT INTO knowledge_base (title, content, embedding) VALUES (?, ?, ?)',
                (title, content, embedding_bytes)
            )
            conn.commit()

    def get_all_knowledge(self) -> List[Tuple[str, str, np.ndarray]]:
        """Retrieve all knowledge base entries with their embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT title, content, embedding FROM knowledge_base')
            results = []
            for row in cursor.fetchall():
                title, content, embedding_bytes = row
                embedding = pickle.loads(embedding_bytes)
                results.append((title, content, embedding))
            return results

    def clear_knowledge_base(self):
        """Clear all entries from the knowledge base."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM knowledge_base')
            conn.commit()

    def get_knowledge_count(self) -> int:
        """Get the number of entries in the knowledge base."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM knowledge_base')
            return cursor.fetchone()[0] 