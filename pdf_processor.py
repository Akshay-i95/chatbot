"""
PDF Stream Processor - Intelligent streaming and parsing of PDF files from Azure Blob Storage

This module handles:
- Streaming PDFs from Azure Blob Storage without downloading
- Concurrent processing with intelligent batching
- Memory-efficient text extraction
- Chunking with overlap for better context
"""

import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Generator, Tuple, Optional
import numpy as np
from tqdm import tqdm

# PDF Processing
import PyPDF2
import pdfplumber

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

# Text Processing
import tiktoken
import re

class PDFStreamProcessor:
    def __init__(self, config: Dict):
        try:
            self.config = config
            
            # Validate and set configuration with proper error handling
            self.max_workers = max(1, min(int(config.get('max_workers', 8)), 16))  # Limit workers
            self.chunk_size = max(100, int(config.get('chunk_size', 1000)))
            self.chunk_overlap = max(0, min(int(config.get('chunk_overlap', 200)), self.chunk_size // 2))
            self.batch_size = max(1, int(config.get('batch_size', 50)))
            self.max_memory_mb = max(512, int(config.get('max_memory_mb', 2048)))
            self.min_chunk_length = max(50, int(config.get('min_chunk_length', 100)))
            self.max_chunk_length = max(self.chunk_size, int(config.get('max_chunk_length', 2000)))
            
            # Initialize tokenizer with error handling
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load tiktoken, using basic chunking: {str(e)}")
                self.tokenizer = None
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            
            # Validate configuration
            self._validate_config()
            
        except Exception as e:
            raise ValueError(f"Failed to initialize PDFStreamProcessor: {str(e)}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.min_chunk_length > self.max_chunk_length:
            raise ValueError("min_chunk_length must be less than max_chunk_length")
        
        self.logger.info(f"✅ PDFStreamProcessor initialized with {self.max_workers} workers, batch size {self.batch_size}")
        
    def estimate_memory_usage(self, file_size_mb: float) -> float:
        """Estimate memory usage for processing a PDF file"""
        # Rough estimate: 3x file size for processing overhead
        return file_size_mb * 3
    
    def stream_pdf_from_blob(self, container_client, blob_name: str) -> Tuple[Optional[io.BytesIO], Optional[Dict]]:
        """Stream PDF content from Azure Blob Storage without downloading to disk"""
        if not blob_name or not blob_name.lower().endswith('.pdf'):
            self.logger.error(f"❌ Invalid PDF file name: {blob_name}")
            return None, None
            
        try:
            blob_client = container_client.get_blob_client(blob_name)
            
            # Get blob properties first with timeout
            try:
                properties = blob_client.get_blob_properties()
            except Exception as e:
                self.logger.error(f"❌ Failed to get properties for {blob_name}: {str(e)}")
                return None, None
            
            file_size_mb = properties.size / (1024 * 1024)
            
            # Validate file size
            if properties.size == 0:
                self.logger.warning(f"⚠️ File {blob_name} is empty")
                return None, None
            
            if file_size_mb > 100:  # Limit to 100MB per file
                self.logger.warning(f"⚠️ File {blob_name} ({file_size_mb:.1f}MB) exceeds size limit")
                return None, None
            
            # Check memory constraints
            estimated_memory = self.estimate_memory_usage(file_size_mb)
            if estimated_memory > self.max_memory_mb:
                self.logger.warning(f"⚠️ File {blob_name} ({file_size_mb:.1f}MB) may exceed memory limit")
                return None, None
            
            # Stream the blob content with timeout and retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    blob_data = blob_client.download_blob()
                    pdf_stream = io.BytesIO()
                    
                    # Stream in chunks to manage memory
                    chunk_size = 8192  # 8KB chunks
                    total_read = 0
                    
                    for chunk in blob_data.chunks():
                        if not chunk:
                            break
                        pdf_stream.write(chunk)
                        total_read += len(chunk)
                        
                        # Safety check for runaway downloads
                        if total_read > 100 * 1024 * 1024:  # 100MB limit
                            raise Exception("File size exceeded during download")
                    
                    pdf_stream.seek(0)
                    
                    # Verify we got a valid PDF stream
                    if pdf_stream.getvalue()[:4] != b'%PDF':
                        raise Exception("Downloaded content is not a valid PDF")
                    
                    return pdf_stream, properties
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"⚠️ Attempt {attempt + 1} failed for {blob_name}: {str(e)}, retrying...")
                        time.sleep(1)
                    else:
                        raise e
            
        except AzureError as e:
            self.logger.error(f"❌ Azure error streaming {blob_name}: {str(e)}")
            return None, None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error streaming {blob_name}: {str(e)}")
            return None, None
    
    def extract_text_from_pdf_stream(self, pdf_stream: io.BytesIO, blob_name: str) -> Tuple[str, Dict]:
        """Extract text from PDF stream using multiple methods for robustness"""
        text_content = ""
        metadata = {
            'filename': blob_name,
            'pages': 0,
            'extraction_method': 'none',
            'processing_time': 0,
            'file_size_mb': 0
        }
        
        start_time = time.time()
        
        try:
            # Method 1: Try pdfplumber first (better text extraction)
            pdf_stream.seek(0)
            with pdfplumber.open(pdf_stream) as pdf:
                pages_text = []
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error extracting page {page_num} from {blob_name}: {str(e)}")
                
                if pages_text:
                    text_content = "\n\n".join(pages_text)
                    metadata['pages'] = len(pdf.pages)
                    metadata['extraction_method'] = 'pdfplumber'
                    
        except Exception as e:
            self.logger.warning(f"⚠️ pdfplumber failed for {blob_name}: {str(e)}")
            
            # Method 2: Fallback to PyPDF2
            try:
                pdf_stream.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                pages_text = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error extracting page {page_num} from {blob_name}: {str(e)}")
                
                if pages_text:
                    text_content = "\n\n".join(pages_text)
                    metadata['pages'] = len(pdf_reader.pages)
                    metadata['extraction_method'] = 'PyPDF2'
                    
            except Exception as e:
                self.logger.error(f"❌ Both extraction methods failed for {blob_name}: {str(e)}")
        
        # Clean and normalize text
        if text_content:
            text_content = self.clean_text(text_content)
        
        metadata['processing_time'] = time.time() - start_time
        metadata['character_count'] = len(text_content)
        
        return text_content, metadata
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)
        
        # Remove very short lines (likely formatting artifacts)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines).strip()
    
    def create_intelligent_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Create intelligent text chunks with overlap and context preservation"""
        if not text or len(text.strip()) < self.min_chunk_length:
            return []
        
        chunks = []
        text = text.strip()
        
        try:
            # Split by paragraphs first to preserve context
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                return []
            
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                try:
                    # Calculate tokens with fallback
                    if self.tokenizer:
                        para_tokens = len(self.tokenizer.encode(para))
                    else:
                        para_tokens = len(para) // 4  # Rough estimate
                    
                    # If paragraph alone exceeds chunk size, split it
                    if para_tokens > self.chunk_size:
                        # Save current chunk if it has content
                        if current_chunk and len(current_chunk) >= self.min_chunk_length:
                            chunks.append(self.create_chunk_metadata(current_chunk, metadata, len(chunks)))
                            current_chunk = ""
                            current_tokens = 0
                        
                        # Split large paragraph into smaller chunks
                        sentences = self._split_into_sentences(para)
                        temp_chunk = ""
                        temp_tokens = 0
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            if self.tokenizer:
                                sentence_tokens = len(self.tokenizer.encode(sentence))
                            else:
                                sentence_tokens = len(sentence) // 4
                            
                            if temp_tokens + sentence_tokens > self.chunk_size and temp_chunk:
                                if len(temp_chunk) >= self.min_chunk_length:
                                    chunks.append(self.create_chunk_metadata(temp_chunk, metadata, len(chunks)))
                                
                                # Add overlap
                                overlap_text = temp_chunk[-self.chunk_overlap:] if len(temp_chunk) > self.chunk_overlap else temp_chunk
                                temp_chunk = overlap_text + ". " + sentence
                                temp_tokens = len(self.tokenizer.encode(temp_chunk)) if self.tokenizer else len(temp_chunk) // 4
                            else:
                                temp_chunk += ". " + sentence if temp_chunk else sentence
                                temp_tokens += sentence_tokens
                        
                        if temp_chunk and len(temp_chunk) >= self.min_chunk_length:
                            chunks.append(self.create_chunk_metadata(temp_chunk, metadata, len(chunks)))
                    
                    # Normal paragraph processing
                    elif current_tokens + para_tokens > self.chunk_size and current_chunk:
                        if len(current_chunk) >= self.min_chunk_length:
                            chunks.append(self.create_chunk_metadata(current_chunk, metadata, len(chunks)))
                        
                        # Add overlap from previous chunk
                        if chunks and current_chunk:
                            overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                            current_chunk = overlap_text + "\n\n" + para
                            current_tokens = len(self.tokenizer.encode(current_chunk)) if self.tokenizer else len(current_chunk) // 4
                        else:
                            current_chunk = para
                            current_tokens = para_tokens
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                        current_tokens += para_tokens
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing paragraph: {str(e)}")
                    continue
            
            # Add the last chunk
            if current_chunk and len(current_chunk) >= self.min_chunk_length:
                chunks.append(self.create_chunk_metadata(current_chunk, metadata, len(chunks)))
            
            # Validate chunks
            valid_chunks = []
            for chunk in chunks:
                if (chunk.get('text') and 
                    len(chunk['text'].strip()) >= self.min_chunk_length and
                    len(chunk['text']) <= self.max_chunk_length):
                    valid_chunks.append(chunk)
            
            return valid_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error creating chunks: {str(e)}")
            return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        try:
            # Simple sentence splitting with multiple delimiters
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback to word-based splitting
            words = text.split()
            sentences = []
            current = []
            for word in words:
                current.append(word)
                if len(' '.join(current)) > 100:  # Rough sentence length
                    sentences.append(' '.join(current))
                    current = []
            if current:
                sentences.append(' '.join(current))
            return sentences
    
    def create_chunk_metadata(self, chunk_text: str, file_metadata: Dict, chunk_index: int) -> Dict:
        """Create comprehensive metadata for a text chunk"""
        return {
            'text': chunk_text,
            'chunk_index': chunk_index,
            'filename': file_metadata['filename'],
            'file_pages': file_metadata['pages'],
            'extraction_method': file_metadata['extraction_method'],
            'chunk_length': len(chunk_text),
            'chunk_tokens': len(self.tokenizer.encode(chunk_text)),
            'file_processing_time': file_metadata['processing_time']
        }
    
    def process_single_pdf(self, container_client, blob_name: str) -> List[Dict]:
        """Process a single PDF file and return chunks with metadata"""
        try:
            # Stream PDF from blob
            pdf_stream, blob_properties = self.stream_pdf_from_blob(container_client, blob_name)
            if not pdf_stream:
                return []
            
            # Extract text
            text_content, metadata = self.extract_text_from_pdf_stream(pdf_stream, blob_name)
            if not text_content:
                self.logger.warning(f"⚠️ No text extracted from {blob_name}")
                return []
            
            # Add blob properties to metadata
            metadata['file_size_mb'] = blob_properties.size / (1024 * 1024)
            metadata['last_modified'] = blob_properties.last_modified
            
            # Create intelligent chunks
            chunks = self.create_intelligent_chunks(text_content, metadata)
            
            self.logger.info(f"✅ Processed {blob_name}: {len(chunks)} chunks, {metadata['pages']} pages")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {blob_name}: {str(e)}")
            return []
    
    def process_pdf_batch(self, container_client, pdf_files: List[str]) -> List[Dict]:
        """Process a batch of PDF files concurrently"""
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_pdf, container_client, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Collect results with progress bar
            with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_file):
                    pdf_file = future_to_file[future]
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                        pbar.set_postfix({'chunks': len(all_chunks)})
                    except Exception as e:
                        self.logger.error(f"❌ Error processing {pdf_file}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        return all_chunks
    
    def process_all_pdfs(self, container_client, pdf_files: List[str]) -> Generator[List[Dict], None, None]:
        """Process all PDF files in intelligent batches"""
        total_files = len(pdf_files)
        self.logger.info(f"🚀 Starting processing of {total_files} PDF files")
        self.logger.info(f"⚙️ Configuration: {self.max_workers} workers, batch size {self.batch_size}")
        
        # Process in batches
        for i in range(0, total_files, self.batch_size):
            batch = pdf_files[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            start_time = time.time()
            chunks = self.process_pdf_batch(container_client, batch)
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ Batch {batch_num} completed: {len(chunks)} chunks in {processing_time:.1f}s")
            
            yield chunks
