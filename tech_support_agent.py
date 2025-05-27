import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_store import KnowledgeStore
import re
from difflib import SequenceMatcher

class TechSupportAgent:
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        """Initialize the tech support agent with a knowledge base directory."""
        self.knowledge_base_dir = knowledge_base_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_store = KnowledgeStore()
        self.load_knowledge_base()

    def load_knowledge_base(self) -> None:
        """Load and index all text files from the knowledge base directory."""
        if not os.path.exists(self.knowledge_base_dir):
            os.makedirs(self.knowledge_base_dir)
            print(f"Created knowledge base directory: {self.knowledge_base_dir}")
            return

        print("Learning from knowledge base files...")
        # Get list of files in knowledge base
        current_files = set(os.listdir(self.knowledge_base_dir))
        
        # Get list of files already in database
        stored_files = set(entry[0] for entry in self.knowledge_store.get_all_knowledge())
        
        # Find new files to process
        new_files = current_files - stored_files
        
        if new_files:
            print(f"Found {len(new_files)} new files to process...")
            for filename in new_files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.knowledge_base_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            # Create embeddings for the content
                            embedding = self.model.encode(content)
                            # Store in database
                            self.knowledge_store.store_knowledge(filename, content, embedding)
                            print(f"Processed: {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            print("Knowledge base update complete!")
        else:
            print("No new files to process.")

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common words and punctuation
        text = text.lower()
        # Split into words and remove short words
        words = [word for word in re.findall(r'\b\w+\b', text) if len(word) > 2]
        return words

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def find_similar_solutions(self, query: str, threshold: float = 0.2) -> List[Tuple[str, float]]:
        """Find similar solutions based on the user's query."""
        # Get all knowledge from store
        knowledge_entries = self.knowledge_store.get_all_knowledge()
        if not knowledge_entries:
            return []

        # Extract keywords from query
        query_keywords = self.extract_keywords(query)
        query_lower = query.lower()
        
        # Encode the query
        query_embedding = self.model.encode(query)
        
        # Calculate similarities using multiple methods
        similarities = []
        for title, content, embedding in knowledge_entries:
            content_lower = content.lower()
            
            # Calculate semantic similarity
            semantic_similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            
            # Calculate text similarity
            text_similarity = self.calculate_text_similarity(query, content)
            
            # Calculate keyword matches
            content_keywords = self.extract_keywords(content)
            keyword_matches = sum(1 for kw in query_keywords if any(
                self.calculate_text_similarity(kw, ckw) > 0.8 for ckw in content_keywords
            ))
            keyword_score = keyword_matches / max(len(query_keywords), 1)
            
            # Calculate exact phrase match score
            exact_phrase_score = 0.0
            for keyword in query_keywords:
                if keyword in content_lower:
                    exact_phrase_score += 1.0
            exact_phrase_score = exact_phrase_score / max(len(query_keywords), 1)
            
            # Calculate title relevance
            title_similarity = self.calculate_text_similarity(query, title)
            
            # Combine scores with adjusted weights
            combined_score = (
                0.25 * semantic_similarity +    # Semantic similarity weight
                0.20 * text_similarity +        # Text similarity weight
                0.20 * keyword_score +          # Keyword matching weight
                0.25 * exact_phrase_score +     # Exact phrase matching weight
                0.10 * title_similarity         # Title relevance weight
            )
            
            if combined_score >= threshold:
                similarities.append((title, combined_score))

        # Sort by similarity score in descending order
        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def get_solution(self, filename: str) -> str:
        """Retrieve the solution content for a specific file."""
        knowledge_entries = self.knowledge_store.get_all_knowledge()
        for title, content, _ in knowledge_entries:
            if title == filename:
                return content
        return "Solution not found."

    def process_query(self, query: str) -> None:
        """Process a user query and provide relevant solutions."""
        print("\nProcessing your query...")
        similar_solutions = self.find_similar_solutions(query)

        if not similar_solutions:
            print("No relevant solutions found. Please try rephrasing your query.")
            return

        print("\nFound the following relevant topics:")
        for i, (filename, score) in enumerate(similar_solutions, 1):
            print(f"{i}. {filename} (Relevance: {score:.2f})")

        if len(similar_solutions) > 1:
            try:
                choice = int(input("\nPlease select a topic number (or 0 to exit): "))
                if choice == 0:
                    return
                if 1 <= choice <= len(similar_solutions):
                    selected_file = similar_solutions[choice - 1][0]
                    print("\nSolution:")
                    print(self.get_solution(selected_file))
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
        else:
            print("\nSolution:")
            print(self.get_solution(similar_solutions[0][0]))

def main():
    agent = TechSupportAgent()
    
    print("Welcome to the Tech Support Agent!")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nPlease describe your technical issue: ")
        if query.lower() == 'exit':
            break
            
        agent.process_query(query)

if __name__ == "__main__":
    main() 