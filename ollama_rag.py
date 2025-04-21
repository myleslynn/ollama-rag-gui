#!/usr/bin/env python3
"""
Ollama RAG System for Markdown Files
-----------------------------------
This script implements a Retrieval-Augmented Generation (RAG) system for Ollama.
It loads markdown files, processes them into chunks, creates embeddings, and allows
querying the knowledge base with Ollama models.
"""

import os
import sys
import json
import glob
import argparse
import logging
import textwrap
from typing import List, Dict, Any

import requests
import numpy as np
from markdown import markdown
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Ollama API settings
OLLAMA_API_BASE = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"  # Default embedding model in Ollama

class Document:
    """Represents a document with text content and metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.embedding = None
    
    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class OllamaRAG:
    """RAG implementation for Ollama."""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL, llm_model: str = "llama3"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.documents = []
        self.embeddings = []
        
        # Check if Ollama is available
        self._check_ollama_availability()
    
    def _check_ollama_availability(self):
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/version")
            if response.status_code == 200:
                version = response.json().get("version", "unknown")
                logger.info(f"Ollama is available (version: {version})")
            else:
                logger.error("Ollama is not responding properly")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
            sys.exit(1)
    
    def _remove_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting by converting to HTML and then extracting text."""
        html = markdown(text)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks of specified size with overlap."""
        # Normalize whitespace and break into sentences
        text = " ".join(text.split())
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Try to manually load punkt if needed
            nltk.download('punkt')
            sentences = text.split('. ')  # Fallback to basic splitting if tokenizer fails
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Add the current chunk to our list of chunks
                chunks.append(" ".join(current_chunk))
                
                # Keep some sentences for overlap
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    overlap_size += len(s)
                    if overlap_size >= overlap:
                        break
                    overlap_sentences.insert(0, s)
                
                # Start a new chunk with overlap
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def load_document_files(self, directory_or_files: str, max_chunk_size: int = 1000):
        """Load document files (markdown, txt) from a directory or specific file pattern."""
        file_paths = []
        
        if os.path.isdir(directory_or_files):
            file_paths = glob.glob(os.path.join(directory_or_files, "**/*.md"), recursive=True)
            # Also include txt files
            file_paths.extend(glob.glob(os.path.join(directory_or_files, "**/*.txt"), recursive=True))
        else:
            file_paths = glob.glob(directory_or_files, recursive=True)
        
        if not file_paths:
            logger.warning(f"No markdown or text files found in {directory_or_files}")
            return
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get relative path as document ID
                rel_path = os.path.relpath(file_path)
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()
                
                # Process content based on file type
                if file_ext == '.md':
                    # Process markdown files
                    clean_text = self._remove_markdown_formatting(content)
                else:
                    # For txt files, use content directly
                    clean_text = content
                
                # Chunk the document
                chunks = self._chunk_text(clean_text, max_chunk_size=max_chunk_size)
                
                logger.info(f"Loaded {file_path} and split into {len(chunks)} chunks")
                
                # Create documents for each chunk
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        metadata={
                            "source": rel_path,
                            "file_name": file_name,
                            "file_type": file_ext,
                            "chunk_id": i
                        }
                    )
                    self.documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} document chunks in total")
    
    # Add backward compatibility alias for the renamed method
    load_markdown_files = load_document_files
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama."""
        try:
            response = requests.post(
                f"{OLLAMA_API_BASE}/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            
            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                logger.error(f"Error getting embedding: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Exception during embedding: {e}")
            return []
    
    def create_embeddings(self):
        """Create embeddings for all documents."""
        if not self.documents:
            logger.warning("No documents to create embeddings for")
            return
        
        logger.info(f"Creating embeddings for {len(self.documents)} documents using model {self.embedding_model}")
        
        for i, doc in enumerate(self.documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(self.documents)}")
            
            embedding = self._get_embedding(doc.content)
            if embedding:
                doc.embedding = embedding
            else:
                logger.warning(f"Failed to get embedding for document {i}")
        
        # Remove documents without embeddings
        self.documents = [doc for doc in self.documents if doc.embedding is not None]
        logger.info(f"Created embeddings for {len(self.documents)} documents")
    
    def save_to_json(self, output_file: str):
        """Save the document store to a JSON file."""
        if not self.documents:
            logger.warning("No documents to save")
            return
        
        data = {
            "embedding_model": self.embedding_model,
            "documents": []
        }
        
        for doc in self.documents:
            data["documents"].append({
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data['documents'])} documents to {output_file}")
    
    def load_from_json(self, input_file: str):
        """Load the document store from a JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the data is in the new format
            if isinstance(data, dict) and "embedding_model" in data:
                self.embedding_model = data["embedding_model"]
                documents_data = data["documents"]
                logger.info(f"Using embedding model from index file: {self.embedding_model}")
            else:
                # Handle old format for backward compatibility
                documents_data = data
                logger.warning("Using default embedding model as index file doesn't specify one")
            
            self.documents = []
            for item in documents_data:
                doc = Document(
                    content=item["content"],
                    metadata=item["metadata"]
                )
                doc.embedding = item["embedding"]
                self.documents.append(doc)
            
            logger.info(f"Loaded {len(self.documents)} documents from {input_file}")
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Search for the most relevant documents to the query."""
        if not self.documents:
            logger.warning("No documents in the store to search")
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.error("Failed to get embedding for query")
            return []
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in similarities[:top_k]]
    
    def generate_response(self, query: str, prompt_template: str = None):
        """Generate a response for the query using RAG."""
        # Search for relevant documents
        search_results = self.search(query)
        
        if not search_results:
            logger.warning("No relevant documents found for the query")
            context = "No relevant information found."
        else:
            # Format the context
            context_parts = []
            for i, (doc, score) in enumerate(search_results, 1):
                source = doc.metadata.get("source", "unknown")
                context_parts.append(
                    f"[Document {i} (from {source}, similarity: {score:.3f})]\n{doc.content}\n"
                )
            
            context = "\n".join(context_parts)
        
        # Default prompt template if none is provided
        if not prompt_template:
            prompt_template = """
            You are a helpful assistant with access to the following information:
            
            {context}
            
            Answer the following question based on the information above:
            {query}
            
            If the information to answer the question is not contained in the provided documents, say so clearly.
            """
        
        # Format the prompt
        prompt = prompt_template.format(context=context, query=query)
        
        try:
            response = requests.post(
                f"{OLLAMA_API_BASE}/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Error generating response: {response.text}")
                return "Error generating response"
        except Exception as e:
            logger.error(f"Exception during response generation: {e}")
            return f"Error: {str(e)}"


def main():
    """Main function to run the Ollama RAG system."""
    parser = argparse.ArgumentParser(description="Ollama RAG for Document Files")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index document files")
    index_parser.add_argument("--input", "-i", required=True, help="Input directory or file pattern")
    index_parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    index_parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size")
    index_parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Embedding model to use")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("--index", "-i", required=True, help="Index JSON file")
    query_parser.add_argument("--query", "-q", help="Query text")
    query_parser.add_argument("--model", "-m", default="llama3", help="LLM model to use")
    query_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.command == "index":
        # Create and index documents
        rag = OllamaRAG(embedding_model=args.embedding_model)
        rag.load_document_files(args.input, max_chunk_size=args.chunk_size)
        rag.create_embeddings()
        rag.save_to_json(args.output)
        logger.info(f"Indexing completed. Output saved to {args.output}")
    
    elif args.command == "query":
        rag = OllamaRAG(llm_model=args.model)
        rag.load_from_json(args.index)
        
        if args.interactive:
            print("\nOllama RAG Interactive Mode")
            print("Type 'exit' or 'quit' to exit\n")
            
            while True:
                query = input("\nEnter your question: ")
                if query.lower() in ("exit", "quit"):
                    break
                
                print("\nSearching for relevant information...")
                response = rag.generate_response(query)
                
                # Pretty print the response
                print("\n" + "=" * 80)
                print("Response:")
                print("-" * 80)
                formatted_response = textwrap.fill(response, width=78)
                print(formatted_response)
                print("=" * 80)
        
        elif args.query:
            response = rag.generate_response(args.query)
            print(response)
        
        else:
            logger.error("Either --query or --interactive must be specified")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 