import os
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional
import magic
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging

class AdvancedDocumentProcessor:
    """
    High-performance, secure document processing service with advanced optimization.
    """
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 max_workers: int = None,
                 cache_size: int = 128):
        """
        Initialize the advanced document processor.
        
        Args:
            model_name (str): Embedding model to use
            max_workers (int): Maximum number of workers for parallel processing
            cache_size (int): LRU cache size for memoization
        """
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or (os.cpu_count() or 1) * 2
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Performance optimization configurations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    @lru_cache(maxsize=128)
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type using magic library with caching.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            str: Detected file type
        """
        try:
            return magic.from_file(file_path)
        except Exception as e:
            self.logger.error(f"File type detection failed: {e}")
            return "unknown"
    
    def _secure_content_extraction(self, file_path: str) -> str:
        """
        Secure content extraction with file size and type validation.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            str: Extracted content
        """
        # File size validation
        max_file_size = 100 * 1024 * 1024  # 100 MB
        if os.path.getsize(file_path) > max_file_size:
            raise ValueError("File size exceeds maximum allowed limit")
        
        # Content extraction based on file type
        file_type = self.detect_file_type(file_path)
        
        extractors = {
            'PDF': self._extract_pdf_content,
            'text': self._extract_text_content,
            'Microsoft Word': self._extract_docx_content
        }
        
        extractor = extractors.get(file_type, self._extract_text_content)
        return extractor(file_path)
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """Extract content from PDF files."""
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() for page in reader.pages)
    
    def _extract_text_content(self, file_path: str) -> str:
        """Extract content from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_docx_content(self, file_path: str) -> str:
        """Extract content from Microsoft Word files."""
        from docx import Document
        doc = Document(file_path)
        return " ".join(para.text for para in doc.paragraphs)
    
    def chunk_content(self, 
                      content: str, 
                      chunk_size: int = 512, 
                      overlap: int = 128) -> List[str]:
        """
        Intelligent content chunking with overlap.
        
        Args:
            content (str): Input text content
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
        
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(content)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
        
        return chunks
    
    def vectorize_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Vectorize text chunks using a pre-trained model.
        
        Args:
            chunks (List[str]): Text chunks to vectorize
        
        Returns:
            numpy array of embeddings
        """
        with torch.no_grad():
            embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors='pt', 
                                        max_length=512, 
                                        truncation=True).to(self.device)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding)
            
            return np.concatenate(embeddings)
    
    async def process_document(self, 
                                file_path: str, 
                                chunk_size: int = 512, 
                                overlap: int = 128) -> Dict[str, Any]:
        """
        Asynchronous document processing with parallel optimization.
        
        Args:
            file_path (str): Path to the document
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
        
        Returns:
            Processing results and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Secure content extraction
            content = await asyncio.to_thread(self._secure_content_extraction, file_path)
            
            # Chunking
            chunks = await asyncio.to_thread(
                self.chunk_content, 
                content, 
                chunk_size, 
                overlap
            )
            
            # Vectorization
            vectors = await asyncio.to_thread(self.vectorize_chunks, chunks)
            
            # Compute document hash for integrity
            document_hash = hashlib.sha256(content.encode()).hexdigest()
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "status": "processed",
                "chunks": len(chunks),
                "document_hash": document_hash,
                "processing_time": processing_time,
                "vectors": vectors
            }
        
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def batch_process_documents(self, 
                                file_paths: List[str], 
                                chunk_size: int = 512, 
                                overlap: int = 128) -> List[Dict[str, Any]]:
        """
        Batch document processing with parallel execution.
        
        Args:
            file_paths (List[str]): List of document file paths
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
        
        Returns:
            List of processing results
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_path = {
                executor.submit(
                    asyncio.run, 
                    self.process_document(path, chunk_size, overlap)
                ): path for path in file_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    result['file_path'] = path
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Processing failed for {path}: {e}")
        
        return results

# Performance monitoring and logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Example usage and performance demonstration.
    """
    processor = AdvancedDocumentProcessor()
    
    # Example document processing
    test_files = [
        '/path/to/document1.pdf',
        '/path/to/document2.docx',
        '/path/to/document3.txt'
    ]
    
    results = processor.batch_process_documents(test_files)
    
    for result in results:
        print(f"Processed {result.get('file_path', 'Unknown')}: {result['status']}")
        if result['status'] == 'processed':
            print(f"  Chunks: {result['chunks']}")
            print(f"  Processing Time: {result['processing_time']:.4f} seconds")

if __name__ == "__main__":
    main()