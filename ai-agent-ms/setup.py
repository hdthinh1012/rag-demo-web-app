# Section 1: Import necessary libraries
import os
import asyncio
import time
import hashlib
import pickle
import json
from typing import Any, Dict, List, Optional
import PyPDF2
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Section 1.5: Persistent Storage Manager
class VectorDBPersistenceManager:
    """Manages persistent storage and file change detection for vector database"""
    
    def __init__(self, storage_dir: str = "./vector_db_storage", upload_folder: str = "./files-db"):
        self.storage_dir = storage_dir
        self.upload_folder = upload_folder
        self.vector_db_file = os.path.join(storage_dir, "vector_db.pkl")
        self.metadata_file = os.path.join(storage_dir, "file_metadata.json")
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {str(e)}")
            return ""
    
    def get_file_metadata(self, file_path: str) -> Dict:
        """Get file metadata including hash, size, and modification time"""
        try:
            stat = os.stat(file_path)
            return {
                'hash': self.get_file_hash(file_path),
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'path': file_path
            }
        except Exception as e:
            print(f"Error getting metadata for {file_path}: {str(e)}")
            return {}
    
    def save_metadata(self, file_metadata: Dict[str, Dict]):
        """Save file metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(file_metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
    
    def load_metadata(self) -> Dict[str, Dict]:
        """Load file metadata from JSON file"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
        return {}
    
    def save_vector_db(self, vector_db: pd.DataFrame):
        """Save vector database to pickle file"""
        try:
            with open(self.vector_db_file, 'wb') as f:
                pickle.dump(vector_db, f)
            print(f"✅ Vector database saved to {self.vector_db_file}")
        except Exception as e:
            print(f"Error saving vector database: {str(e)}")
            raise
    
    def load_vector_db(self) -> Optional[pd.DataFrame]:
        """Load vector database from pickle file"""
        try:
            if os.path.exists(self.vector_db_file):
                with open(self.vector_db_file, 'rb') as f:
                    vector_db = pickle.load(f)
                print(f"✅ Vector database loaded from {self.vector_db_file}")
                print(f"📊 Loaded {len(vector_db)} chunks from {vector_db['document_name'].nunique()} documents")
                return vector_db
        except Exception as e:
            print(f"Error loading vector database: {str(e)}")
        return None
    
    def detect_file_changes(self, current_files: List[str]) -> Dict[str, List[str]]:
        """Detect which files have been added, modified, or removed"""
        old_metadata = self.load_metadata()
        current_metadata = {}
        
        # Get current file metadata
        for file_path in current_files:
            if os.path.exists(file_path):
                current_metadata[file_path] = self.get_file_metadata(file_path)
        
        # Detect changes
        changes = {
            'added': [],
            'modified': [],
            'removed': [],
            'unchanged': []
        }
        
        # Check for new and modified files
        for file_path, metadata in current_metadata.items():
            if file_path not in old_metadata:
                changes['added'].append(file_path)
            elif metadata.get('hash') != old_metadata[file_path].get('hash'):
                changes['modified'].append(file_path)
            else:
                changes['unchanged'].append(file_path)
        
        # Check for removed files
        for file_path in old_metadata:
            if file_path not in current_metadata:
                changes['removed'].append(file_path)
        
        return changes, current_metadata
    
    def needs_rebuild(self, current_files: List[str]) -> tuple[bool, Dict]:
        """Check if vector database needs to be rebuilt"""
        # If no stored vector database exists, rebuild is needed
        if not os.path.exists(self.vector_db_file):
            print("📋 No existing vector database found - full rebuild needed")
            return True, {'reason': 'no_existing_db'}
        
        # Check for file changes
        changes, current_metadata = self.detect_file_changes(current_files)
        
        total_changes = len(changes['added']) + len(changes['modified']) + len(changes['removed'])
        
        if total_changes > 0:
            print(f"📋 File changes detected:")
            if changes['added']:
                print(f"   • Added: {[os.path.basename(f) for f in changes['added']]}")
            if changes['modified']:
                print(f"   • Modified: {[os.path.basename(f) for f in changes['modified']]}")
            if changes['removed']:
                print(f"   • Removed: {[os.path.basename(f) for f in changes['removed']]}")
            
            return True, {'reason': 'file_changes', 'changes': changes, 'metadata': current_metadata}
        
        print(f"✅ No file changes detected - using existing vector database")
        return False, {'reason': 'no_changes', 'metadata': current_metadata}
    
    def update_incremental(self, vector_db: pd.DataFrame, changes: Dict, 
                          embedding_client: Any, embedding_model: str, 
                          chunk_size: int = 4096) -> pd.DataFrame:
        """Update vector database incrementally based on file changes"""
        
        # Remove chunks from deleted/modified files
        files_to_remove = changes['removed'] + changes['modified']
        if files_to_remove:
            print(f"🗑️ Removing chunks from {len(files_to_remove)} files...")
            vector_db = vector_db[~vector_db['document_name'].isin(files_to_remove)]
        
        # Add chunks for new/modified files
        files_to_add = changes['added'] + changes['modified']
        if files_to_add:
            print(f"➕ Adding chunks from {len(files_to_add)} files...")
            new_chunks_df = build_index(
                files_to_add, 
                embedding_client, 
                embedding_model, 
                chunk_size=chunk_size,
                is_incremental=True
            )
            vector_db = pd.concat([vector_db, new_chunks_df], ignore_index=True)
        
        return vector_db

# Section 2: Setup the GENAI Client
def setup_genai_client(app):
    try:
        """Initialize the GenAI client for Vertex AI"""
        PROJECT_ID = os.getenv("PROJECT_ID", "personal-rag-ai-agent")
        LOCATION = os.getenv("LOCATION", "asia-southeast1")
        
        client = genai.Client(
            vertexai=True,
            credentials=os.getenv("SERVICE_ACCOUNT_CREDENTIALS"),
            project=PROJECT_ID,
            location=LOCATION
        )
        
        MODEL_ID = "gemini-2.5-flash"
        MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
        text_embedding_model = "gemini-embedding-001"

        return client, MODEL, text_embedding_model
    except Exception as e:
        app.logger.error(f"Error setting up GenAI client: {str(e)}")
        raise

# Section 3: Setup the RAG pipeline

# Section 3.1: Setup Embedding Model & Vector Database
@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
def get_embeddings(
    embedding_client: Any, embedding_model: str, text: str, output_dim: int = 768
) -> list[float]:
    """Generate embeddings for text with retry logic for API quota management."""
    try:
        response = embedding_client.models.embed_content(
            model=embedding_model,
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=output_dim),
        )
        return [response.embeddings[0].values]
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return None
        print(f"Error generating embeddings: {str(e)}")
        raise

def build_index(
    document_paths: list[str],
    embedding_client: Any,
    embedding_model: str,
    chunk_size: int = 4096,
    # NEW: Throttling parameters
    throttle_config: dict = None,
    progress_callback = None,
    # NEW: Incremental update flag
    is_incremental: bool = False,
) -> pd.DataFrame:
    """Build searchable index from a list of PDF documents with throttling mechanism."""
    
    # Initialize throttle with default or custom config
    if throttle_config is None:
        throttle_config = {
            'requests_per_minute': 60,  # Google Vertex AI default limit
            'batch_size': 10,           # Process 10 chunks then pause
            'batch_delay': 1.0,         # 1 second pause between batches
        }
    
    throttle = IndexBuildThrottle(
        requests_per_minute=throttle_config.get('requests_per_minute', 60),
        batch_size=throttle_config.get('batch_size', 10),
        batch_delay=throttle_config.get('batch_delay', 1.0),
        progress_callback=progress_callback
    )
    
    all_chunks = []
    total_chunks_estimated = 0
    processed_chunks = 0
    
    # First pass: estimate total chunks for progress tracking
    print("Estimating total chunks for progress tracking...")
    for doc_path in document_paths:
        try:
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    chunks = [
                        page_text[i : i + chunk_size]
                        for i in range(0, len(page_text), chunk_size)
                    ]
                    estimated_chunks = len(chunks) 
                    total_chunks_estimated += estimated_chunks
        except Exception as e:
            print(f"Warning: Could not estimate chunks for {doc_path}: {str(e)}")
    
    action_type = "incremental update" if is_incremental else "full index build"
    print(f"Estimated {total_chunks_estimated} total chunks to process ({action_type})")
    start_time = time.time()

    # Second pass: actual processing with throttling
    for doc_idx, doc_path in enumerate(document_paths):
        try:
            print(f"\nProcessing document {doc_idx + 1}/{len(document_paths)}: {os.path.basename(doc_path)}")
            
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    chunks = [
                        page_text[i : i + chunk_size]
                        for i in range(0, len(page_text), chunk_size)
                    ]

                    for chunk_num, chunk_text in enumerate(chunks):
                        # Apply throttling before API call
                        throttle.wait_if_needed()
                        
                        embeddings = get_embeddings(
                            embedding_client, embedding_model, chunk_text
                        )

                        processed_chunks += 1

                        if embeddings is None:
                            print(f"Warning: Could not generate embeddings for chunk {chunk_num} on page {page_num + 1}")
                            continue

                        chunk_info = {
                            "document_name": doc_path,
                            "page_number": page_num + 1,
                            "page_text": page_text,
                            "chunk_number": chunk_num,
                            "chunk_text": chunk_text,
                            "embeddings": embeddings,
                        }
                        all_chunks.append(chunk_info)
                        
                        # Update progress and apply batch delays
                        throttle.update_progress(processed_chunks, total_chunks_estimated, "chunks")
                        throttle.batch_delay_if_needed(processed_chunks)

        except Exception as e:
            print(f"Error processing document {doc_path}: {str(e)}")
            continue

    if not all_chunks:
        raise ValueError("No chunks were created from the documents")

    elapsed_time = time.time() - start_time
    success_rate = (throttle.successful_requests / throttle.total_requests) * 100 if throttle.total_requests > 0 else 0
    
    print(f"\n✅ Index building completed!")
    print(f"📊 Statistics:")
    print(f"   • Total time: {elapsed_time:.1f}s")
    print(f"   • Chunks processed: {len(all_chunks)}")
    print(f"   • API calls made: {throttle.total_requests}")
    print(f"   • Success rate: {success_rate:.1f}%")
    print(f"   • Average time per chunk: {elapsed_time/len(all_chunks):.2f}s")

    return pd.DataFrame(all_chunks)


# Section 3.1.5: Persistent Storage Functions
def load_or_build_vector_db(
    upload_folder: str,
    embedding_client: Any,
    embedding_model: str,
    storage_dir: str = "./vector_db_storage",
    chunk_size: int = 4096,
    throttle_config: dict = None,
    progress_callback = None,
    force_rebuild: bool = False
) -> Optional[pd.DataFrame]:
    """Load existing vector database or build new one with smart file change detection"""
    
    # Initialize persistence manager
    persistence_manager = VectorDBPersistenceManager(storage_dir, upload_folder)
    
    # Get current PDF files
    import glob
    current_files = glob.glob(os.path.join(upload_folder, "*.pdf"))
    
    if not current_files:
        print("📋 No PDF files found in upload folder")
        return None
    
    print(f"📁 Found {len(current_files)} PDF files in {upload_folder}")
    
    # Check if rebuild is needed (unless forced)
    if not force_rebuild:
        needs_rebuild, rebuild_info = persistence_manager.needs_rebuild(current_files)
        
        if not needs_rebuild:
            # Load existing vector database
            vector_db = persistence_manager.load_vector_db()
            if vector_db is not None:
                return vector_db
            else:
                print("⚠️ Failed to load existing vector database - falling back to full rebuild")
        elif rebuild_info['reason'] == 'file_changes':
            # Try incremental update
            print("🔄 Attempting incremental update...")
            try:
                existing_vector_db = persistence_manager.load_vector_db()
                if existing_vector_db is not None:
                    updated_vector_db = persistence_manager.update_incremental(
                        existing_vector_db,
                        rebuild_info['changes'],
                        embedding_client,
                        embedding_model,
                        chunk_size
                    )
                    
                    # Save updated database and metadata
                    persistence_manager.save_vector_db(updated_vector_db)
                    persistence_manager.save_metadata(rebuild_info['metadata'])
                    
                    print("✅ Incremental update completed successfully!")
                    return updated_vector_db
                else:
                    print("⚠️ Could not load existing database for incremental update - doing full rebuild")
            except Exception as e:
                print(f"⚠️ Incremental update failed: {str(e)} - doing full rebuild")
    
    # Full rebuild
    print("🔄 Building vector database from scratch...")
    try:
        vector_db = build_index(
            current_files,
            embedding_client,
            embedding_model,
            chunk_size=chunk_size,
            throttle_config=throttle_config,
            progress_callback=progress_callback,
            is_incremental=False
        )
        
        # Save the new database and metadata
        persistence_manager.save_vector_db(vector_db)
        
        # Save current file metadata
        current_metadata = {}
        for file_path in current_files:
            current_metadata[file_path] = persistence_manager.get_file_metadata(file_path)
        persistence_manager.save_metadata(current_metadata)
        
        print("✅ Vector database built and saved successfully!")
        return vector_db
        
    except Exception as e:
        print(f"❌ Error building vector database: {str(e)}")
        raise

def get_storage_info(storage_dir: str = "./vector_db_storage") -> Dict:
    """Get information about stored vector database"""
    persistence_manager = VectorDBPersistenceManager(storage_dir)
    
    info = {
        'storage_exists': os.path.exists(persistence_manager.vector_db_file),
        'metadata_exists': os.path.exists(persistence_manager.metadata_file),
        'storage_size': 0,
        'last_modified': None,
        'file_count': 0
    }
    
    if info['storage_exists']:
        try:
            stat = os.stat(persistence_manager.vector_db_file)
            info['storage_size'] = stat.st_size
            info['last_modified'] = stat.st_mtime
            
            # Try to get file count from vector database
            vector_db = persistence_manager.load_vector_db()
            if vector_db is not None:
                info['file_count'] = vector_db['document_name'].nunique()
                info['chunk_count'] = len(vector_db)
        except Exception as e:
            print(f"Error getting storage info: {str(e)}")
    
    return info

# Section 3.2: Setup Fetch Relevant Chunks (Retrieval)
def get_relevant_chunks(
    query: str,
    vector_db: pd.DataFrame,
    embedding_client: Any,
    embedding_model: str,
    top_k: int = 3,
) -> str:
    """Retrieve the most relevant document chunks for a query using similarity search."""
    try:
        query_embedding = get_embeddings(embedding_client, embedding_model, query)

        if query_embedding is None:
            return "Could not process query due to quota issues"

        similarities = [
            cosine_similarity(query_embedding, chunk_emb)[0][0]
            for chunk_emb in vector_db["embeddings"]
        ]

        top_indices = np.argsort(similarities)[-top_k:]
        relevant_chunks = vector_db.iloc[top_indices]

        context = []
        for _, row in relevant_chunks.iterrows():
            context.append(
                {
                    "document_name": row["document_name"],
                    "page_number": row["page_number"],
                    "chunk_number": row["chunk_number"],
                    "chunk_text": row["chunk_text"],
                }
            )

        return "\n\n".join(
            [
                f"[Page {chunk['page_number']}, Chunk {chunk['chunk_number']}]: {chunk['chunk_text']}"
                for chunk in context
            ]
        )

    except Exception as e:
        print(f"Error getting relevant chunks: {str(e)}")
        return "Error retrieving relevant chunks"

# Section 3.3: Setup Augmented and Generation Audio Response
async def generate_audio_content(query: str, client: Any, model: str):
    """Function to generate audio response for provided query using Gemini Multimodal Live API."""
    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
            "language_code": "en-US"
        },
    }

    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send(input=query, end_of_turn=True)

        audio_parts = []
        async for message in session.receive():
            if message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data:
                        audio_parts.append(
                            np.frombuffer(part.inline_data.data, dtype=np.int16)
                        )

            if message.server_content.turn_complete:
                if audio_parts:
                    audio_data = np.concatenate(audio_parts, axis=0)
                    return audio_data
                break
        return None

@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
async def generate_answer_with_audio(
    query: str, context: str, llm_client: Any, model: str
) -> np.ndarray:
    """Generate audio answer using LLM with retry logic for API quota management."""
    try:
        if context in [
            "Could not process query due to quota issues",
            "Error retrieving relevant chunks",
        ]:
            return None

        prompt = f"""Based on the following context, please answer the question.

        Context:
        {context}

        Question: {query}

        Answer:"""

        audio_data = await generate_audio_content(prompt, llm_client, model)
        return audio_data

    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return None
        print(f"Error generating answer: {str(e)}")
        return None

# Section 3.4: Setup Text Generation (Alternative to Audio)
def generate_text_content(query: str, client: Any, model: str) -> str:
    """Function to generate text response for provided query using Gemini API."""
    try:
        # Use the standard Gemini API for text generation
        response = client.models.generate_content(
            model=model,
            contents=[query],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.9,
                top_k=40
            )
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            print("Warning: Empty response from Gemini API")
            return "I apologize, but I couldn't generate a response at this time."
            
    except Exception as e:
        print(f"Error generating text content: {str(e)}")
        if "RESOURCE_EXHAUSTED" in str(e):
            return "Service temporarily unavailable due to quota limits. Please try again later."
        return f"Error generating response: {str(e)}"

@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
def generate_answer_with_text(
    query: str, context: str, llm_client: Any, model: str
) -> str:
    """Generate text answer using LLM with retry logic for API quota management."""
    try:
        if context in [
            "Could not process query due to quota issues",
            "Error retrieving relevant chunks",
        ]:
            return "I apologize, but I'm currently unable to process your query due to technical issues. Please try again later."

        prompt = f"""Based on the following context, please answer the question comprehensively and accurately. If not found necessary information in the context, answer with what you have been trained with.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context provided above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing.

Answer:"""

        text_response = generate_text_content(prompt, llm_client, model)
        return text_response

    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "Service temporarily unavailable due to quota limits. Please try again later."
        print(f"Error generating text answer: {str(e)}")
        return f"I apologize, but I encountered an error while generating the response: {str(e)}"

class IndexBuildThrottle:
    """Throttle mechanism for build_index operations"""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 batch_size: int = 10,
                 batch_delay: float = 1.0,
                 progress_callback=None):
        self.requests_per_minute = requests_per_minute
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.progress_callback = progress_callback
        
        # Rate limiting tracking
        self.request_times = []
        self.total_requests = 0
        self.successful_requests = 0
        
    def wait_if_needed(self):
        """Wait if we're exceeding the rate limit"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we're at the rate limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0]) + 0.1
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Record this request
        self.request_times.append(current_time)
        self.total_requests += 1
    
    def batch_delay_if_needed(self, current_item: int):
        """Add delay after processing a batch"""
        if current_item > 0 and current_item % self.batch_size == 0:
            print(f"Processed {current_item} items. Pausing for {self.batch_delay}s...")
            time.sleep(self.batch_delay)
    
    def update_progress(self, current: int, total: int, item_name: str = "items"):
        """Update progress and call callback if provided"""
        self.successful_requests += 1
        progress_pct = (current / total) * 100 if total > 0 else 0
        
        print(f"Progress: {current}/{total} {item_name} ({progress_pct:.1f}%) - "
              f"Success rate: {self.successful_requests}/{self.total_requests}")
        
        if self.progress_callback:
            self.progress_callback(current, total, self.successful_requests, self.total_requests)
