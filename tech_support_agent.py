import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from knowledge_store import KnowledgeStore
import re
from difflib import SequenceMatcher
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

class TechSupportAgent:
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        """Initialize the tech support agent with a knowledge base directory."""
        self.knowledge_base_dir = knowledge_base_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_store = KnowledgeStore()
        self.index = None
        self.knowledge_entries = []
        
        # Load environment variables
        load_dotenv()
        
        # Initialize LangChain components
        try:
            self.llm = Ollama(
                model="llama2",
                temperature=0,
                base_url="http://localhost:11434"
            )
            print("Successfully connected to Llama2")
        except Exception as e:
            print(f"Error connecting to Llama2: {str(e)}")
            print("Please make sure Ollama is running and llama2 is installed")
            raise
        
        # Use HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Load knowledge base
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
            texts = []
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
                            texts.append(content)
                            print(f"Processed: {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            
            # Create vector store with new documents
            if texts:
                self._update_vector_store(texts)
            print("Knowledge base update complete!")
        else:
            print("No new files to process.")
        
        # Initialize FAISS index
        self._initialize_faiss_index()
        
        # Initialize vector store if not exists
        if not self.vector_store:
            self._initialize_vector_store()

    def _update_vector_store(self, texts: List[str]) -> None:
        """Update the vector store with new documents."""
        # Split texts into chunks
        chunks = self.text_splitter.split_text("\n".join(texts))
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings
            )
            # Save the vector store
            self.vector_store.save_local("faiss_index")
        else:
            self.vector_store.add_texts(chunks)
            # Save the updated vector store
            self.vector_store.save_local("faiss_index")
        
        # Create QA chain
        self._create_qa_chain()

    def _initialize_vector_store(self) -> None:
        """Initialize the vector store with existing knowledge base."""
        try:
            # Try to load existing FAISS index
            self.vector_store = FAISS.load_local("faiss_index", self.embeddings)
        except:
            # If no existing index, create new one
            knowledge_entries = self.knowledge_store.get_all_knowledge()
            if not knowledge_entries:
                return
                
            texts = [content for _, content, _ in knowledge_entries]
            self._update_vector_store(texts)

    def _create_qa_chain(self) -> None:
        """Create the QA chain with custom prompt."""
        prompt_template = """You are a technical support assistant. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            print("Successfully created QA chain")
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            raise

    def process_query(self, query: str) -> None:
        """Process a user query and provide relevant solutions."""
        print("\nProcessing your query...")
        
        try:
            # Get similar solutions using FAISS
            similar_solutions = self.find_similar_solutions(query, threshold=0.30)
            
            # Get RAG response
            rag_response = self.qa_chain({"query": query})
            
            # Display results
            print("\nAI-Generated Response:")
            print(rag_response["result"])
            
            print("\nRelevant Knowledge Base Articles:")
            for i, (filename, score) in enumerate(similar_solutions, 1):
                print(f"{i}. {filename} (Relevance: {score:.2f})")

            if len(similar_solutions) > 1:
                try:
                    choice = int(input("\nPlease select a topic number to view details (or 0 to exit): "))
                    if choice == 0:
                        return
                    if 1 <= choice <= len(similar_solutions):
                        selected_file = similar_solutions[choice - 1][0]
                        print("\nDetailed Solution:")
                        print(self.get_solution(selected_file))
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("\nDetailed Solution:")
                print(self.get_solution(similar_solutions[0][0]))
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Please try rephrasing your query or check if Ollama is running properly")

    def _initialize_faiss_index(self) -> None:
        """Initialize FAISS index with current knowledge base."""
        knowledge_entries = self.knowledge_store.get_all_knowledge()
        if not knowledge_entries:
            return

        self.knowledge_entries = knowledge_entries
        embeddings = np.array([entry[2] for entry in knowledge_entries], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)

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
        """Find similar solutions based on the user's query using FAISS."""
        if not self.index or not self.knowledge_entries:
            return []

        # Extract keywords from query
        query_keywords = self.extract_keywords(query)
        query_lower = query.lower()
        
        # Encode the query
        query_embedding = self.model.encode(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search using FAISS
        k = min(10, len(self.knowledge_entries))  # Get top k results
        similarities, indices = self.index.search(query_embedding, k)
        
        # Calculate additional similarity metrics
        results = []
        for idx, semantic_score in zip(indices[0], similarities[0]):
            title, content, _ = self.knowledge_entries[idx]
            content_lower = content.lower()
            
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
                0.25 * float(semantic_score) +      # FAISS semantic similarity weight
                0.20 * text_similarity +            # Text similarity weight
                0.20 * keyword_score +              # Keyword matching weight
                0.25 * exact_phrase_score +         # Exact phrase matching weight
                0.10 * title_similarity             # Title relevance weight
            )
            
            # Only add results that meet or exceed the threshold
            if combined_score >= threshold:
                results.append((title, combined_score))

        # Sort by similarity score in descending order
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_solution(self, filename: str) -> str:
        """Retrieve the solution content for a specific file."""
        knowledge_entries = self.knowledge_store.get_all_knowledge()
        for title, content, _ in knowledge_entries:
            if title == filename:
                return content
        return "Solution not found."

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