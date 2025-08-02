import os
import asyncio
import glob
import time
from typing import Any
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import PyPDF2
import numpy as np
import pandas as pd

# For GenerativeAI
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig

# For similarity score
from sklearn.metrics.pairwise import cosine_similarity

# For retry mechanism
from tenacity import retry, stop_after_attempt, wait_random_exponential

import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = './files-db'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for RAG components
vector_db = None
client = None
executor = ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Import RAG functions from setup.py
from setup import (
    setup_genai_client, 
    build_index, 
    get_relevant_chunks, 
    generate_answer_with_audio,
    load_or_build_vector_db,
    get_storage_info
)

def run_async(coro):
    """Helper function to run async functions in Flask"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def initialize_rag_system(app):
    """Initialize the RAG system components with persistent storage"""
    global client, vector_db, MODEL, text_embedding_model
    
    try:
        # Setup GenAI client
        client, MODEL, text_embedding_model = setup_genai_client(app)
        
        # Configure throttling for initialization
        throttle_config = {
            'requests_per_minute': 50,  # Conservative for startup
            'batch_size': 5,
            'batch_delay': 2.0,
        }
        
        # Load or build vector database with persistent storage
        print("🚀 Initializing RAG system with persistent storage...")
        vector_db = load_or_build_vector_db(
            upload_folder=UPLOAD_FOLDER,
            embedding_client=client,
            embedding_model=text_embedding_model,
            throttle_config=throttle_config
        )
        
        if vector_db is not None:
            print("✅ RAG system initialized successfully!")
            print(f"📊 Loaded {len(vector_db)} chunks from {vector_db['document_name'].nunique()} documents")
        else:
            print("📋 No documents found. RAG system ready for file uploads.")
            
    except Exception as e:
        print(f"❌ Error initializing RAG system: {str(e)}")
        app.logger.error(f"RAG initialization error: {str(e)}", exc_info=True)
        client = None
        vector_db = None

def rebuild_index(force_rebuild: bool = False):
    """Rebuild the vector database index with all PDF files using persistent storage"""
    global vector_db
    
    if not client:
        print("❌ GenAI client not initialized")
        return False
    
    try:
        # Configure throttling for rebuild
        throttle_config = {
            'requests_per_minute': 60,
            'batch_size': 10,
            'batch_delay': 1.0,
        }
        
        # Load or build vector database
        vector_db = load_or_build_vector_db(
            upload_folder=UPLOAD_FOLDER,
            embedding_client=client,
            embedding_model=text_embedding_model,
            throttle_config=throttle_config,
            force_rebuild=force_rebuild
        )
        
        if vector_db is not None:
            print("✅ Index rebuild completed successfully!")
            return True
        else:
            print("❌ No files to process")
            return False
            
    except Exception as e:
        print(f"❌ Error rebuilding index: {str(e)}")
        app.logger.error(f"Index rebuild error: {str(e)}", exc_info=True)
        return False

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    """
    POST endpoint to handle form-data request with query and files
    - query: string - the user's question
    - files: multiple PDF files to upload and process
    """
    app.logger.debug(f"Received request: {request.form}")
    app.logger.debug(f"Files in request: {request.files}")
    app.logger.debug(f"Query: {request.form.get('query')}")
    app.logger.debug(f"Files: {request.files.getlist('files')}")
    
    try:
        # Check if query is provided
        if 'query' not in request.form:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        query = request.form['query']
        app.logger.info(f"Processing query: {query}")
        
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Handle file uploads
        uploaded_files = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    # Add timestamp to avoid filename conflicts
                    timestamp = str(int(time.time() * 1000))
                    filename = f"{timestamp}_{filename}"
                    
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    uploaded_files.append(filepath)
                    print(f"Saved file: {filepath}")
        
        # If new files were uploaded, trigger smart rebuild (incremental update)
        if uploaded_files:
            app.logger.info(f"New files uploaded: {[os.path.basename(f) for f in uploaded_files]}")
            success = rebuild_index(force_rebuild=False)  # Smart rebuild with incremental updates
            if not success:
                return jsonify({'error': 'Failed to update document index'}), 500
        elif vector_db is None:
            # If no vector database exists, try to load or build one
            success = rebuild_index(force_rebuild=False)
            if not success:
                return jsonify({'error': 'Failed to initialize document index'}), 500
        
        # Check if we have a valid vector database
        if vector_db is None or vector_db.empty:
            return jsonify({'error': 'No documents available for processing'}), 400
        
        # Get relevant context
        relevant_context = get_relevant_chunks(
            query, vector_db, client, text_embedding_model, top_k=5
        )
        
        if "Error" in relevant_context or "quota issues" in relevant_context:
            return jsonify({'error': f'Failed to retrieve context: {relevant_context}'}), 500
        
        # Generate audio response using the executor to handle async calls
        future = executor.submit(
            run_async,
            generate_answer_with_audio(query, relevant_context, client, MODEL)
        )
        
        audio_data = future.result(timeout=60)  # 60 second timeout
        
        if audio_data is None:
            return jsonify({'error': 'Failed to generate audio response'}), 500
        
        # Save audio data to temporary file
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Convert numpy array to wav format (simple PCM)
        import wave
        with wave.open(temp_audio_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz sample rate
            wav_file.writeframes(audio_data.tobytes())
        
        # Return the audio file
        return send_file(
            temp_audio_file.name,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'response_{int(time.time())}.wav',
            conditional=False
        )
        app.logger.info(f"Audio file generated successfully: {temp_audio_file.name}")
        
    except Exception as e:
        app.logger.error(f"Error in generate_speech: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with storage information"""
    storage_info = get_storage_info()
    
    status = {
        'status': 'healthy',
        'rag_initialized': client is not None,
        'documents_indexed': vector_db is not None and not vector_db.empty if vector_db is not None else False,
        'upload_folder': UPLOAD_FOLDER,
        'pdf_files_count': len(glob.glob(os.path.join(UPLOAD_FOLDER, "*.pdf"))),
        'storage': {
            'persistent_storage_available': storage_info['storage_exists'],
            'storage_size_bytes': storage_info['storage_size'],
            'indexed_files': storage_info.get('file_count', 0),
            'indexed_chunks': storage_info.get('chunk_count', 0),
            'last_modified': storage_info['last_modified']
        }
    }
    return jsonify(status)

@app.route('/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    pdf_files = glob.glob(os.path.join(UPLOAD_FOLDER, "*.pdf"))
    files_info = []
    
    for file_path in pdf_files:
        file_stat = os.stat(file_path)
        files_info.append({
            'filename': os.path.basename(file_path),
            'size': file_stat.st_size,
            'modified': file_stat.st_mtime
        })
    
    return jsonify({
        'files': files_info,
        'total_count': len(files_info)
    })

@app.route('/storage/info', methods=['GET'])
def storage_info():
    """Get detailed storage information"""
    try:
        storage_info = get_storage_info()
        return jsonify(storage_info)
    except Exception as e:
        return jsonify({'error': f'Failed to get storage info: {str(e)}'}), 500

@app.route('/storage/rebuild', methods=['POST'])
def force_rebuild():
    """Force rebuild of the vector database"""
    try:
        success = rebuild_index(force_rebuild=True)
        if success:
            return jsonify({
                'message': 'Vector database rebuilt successfully',
                'storage_info': get_storage_info()
            })
        else:
            return jsonify({'error': 'Failed to rebuild vector database'}), 500
    except Exception as e:
        app.logger.error(f"Error in force rebuild: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/debug/vector-db-stats', methods=['GET'])
def debug_vector_db_stats():
    """Debug endpoint to get vector database statistics"""
    if not app.debug:
        return jsonify({'error': 'Debug mode only'}), 403
    
    try:
        if vector_db is None:
            return jsonify({'error': 'Vector database not initialized'})
        
        stats = {
            'total_chunks': len(vector_db),
            'unique_documents': vector_db['document_name'].nunique(),
            'documents': vector_db['document_name'].value_counts().to_dict(),
            'average_chunk_length': vector_db['chunk_text'].str.len().mean(),
            'total_pages': vector_db['page_number'].nunique(),
            'memory_usage_mb': vector_db.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500

# if __name__ == '__main__':
print("Initializing RAG system...")
initialize_rag_system(app)
print("Starting Flask server...")
app.run(debug=True, host='0.0.0.0', port=5000)

