#!/usr/bin/env python3
"""
Test script for the persistent storage system
This script demonstrates the functionality without running the full Flask app
"""

import os
import glob
import json
from setup import VectorDBPersistenceManager, get_storage_info

def test_storage_manager():
    """Test the VectorDBPersistenceManager functionality"""
    
    print("🧪 Testing Persistent Storage Manager")
    print("=" * 50)
    
    # Initialize the storage manager
    storage_dir = "./test_vector_db_storage"
    upload_folder = "./files-db"
    
    manager = VectorDBPersistenceManager(storage_dir, upload_folder)
    
    # Test 1: Check current PDF files
    pdf_files = glob.glob(os.path.join(upload_folder, "*.pdf"))
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for file in pdf_files[:5]:  # Show first 5
        print(f"   • {os.path.basename(file)}")
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more files")
    
    # Test 2: Check file metadata
    if pdf_files:
        print(f"\n🔍 Testing file metadata for: {os.path.basename(pdf_files[0])}")
        metadata = manager.get_file_metadata(pdf_files[0])
        print(f"   • Hash: {metadata.get('hash', 'N/A')[:16]}...")
        print(f"   • Size: {metadata.get('size', 0):,} bytes")
        print(f"   • Modified: {metadata.get('mtime', 0)}")
    
    # Test 3: Check if rebuild is needed
    if pdf_files:
        print(f"\n🔄 Checking if rebuild is needed...")
        needs_rebuild, info = manager.needs_rebuild(pdf_files)
        print(f"   • Needs rebuild: {needs_rebuild}")
        print(f"   • Reason: {info.get('reason', 'unknown')}")
        
        if 'changes' in info:
            changes = info['changes']
            print(f"   • Added files: {len(changes.get('added', []))}")
            print(f"   • Modified files: {len(changes.get('modified', []))}")
            print(f"   • Removed files: {len(changes.get('removed', []))}")
            print(f"   • Unchanged files: {len(changes.get('unchanged', []))}")
    
    # Test 4: Check storage info
    print(f"\n📊 Storage Information:")
    storage_info = get_storage_info(storage_dir)
    print(f"   • Storage exists: {storage_info['storage_exists']}")
    print(f"   • Metadata exists: {storage_info['metadata_exists']}")
    print(f"   • Storage size: {storage_info['storage_size']:,} bytes")
    print(f"   • Indexed files: {storage_info.get('file_count', 0)}")
    print(f"   • Indexed chunks: {storage_info.get('chunk_count', 0)}")
    
    # Test 5: Try to load existing vector database
    print(f"\n💾 Attempting to load existing vector database...")
    vector_db = manager.load_vector_db()
    if vector_db is not None:
        print(f"   ✅ Loaded successfully!")
        print(f"   • Total chunks: {len(vector_db)}")
        print(f"   • Unique documents: {vector_db['document_name'].nunique()}")
        print(f"   • Memory usage: {vector_db.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    else:
        print(f"   ℹ️ No existing vector database found")
    
    print(f"\n✅ Storage manager test completed!")

def test_api_endpoints():
    """Test the new API endpoints (requires running Flask server)"""
    
    print("\n🌐 API Endpoints for Testing")
    print("=" * 50)
    
    endpoints = [
        ("GET", "/health", "Health check with storage info"),
        ("GET", "/storage/info", "Detailed storage information"),
        ("POST", "/storage/rebuild", "Force rebuild vector database"),
        ("GET", "/debug/vector-db-stats", "Debug statistics (debug mode only)"),
        ("GET", "/files", "List uploaded files"),
    ]
    
    print("Available endpoints:")
    for method, endpoint, description in endpoints:
        print(f"   • {method:4} {endpoint:20} - {description}")
    
    print(f"\nExample curl commands:")
    print(f"curl http://localhost:5000/health")
    print(f"curl http://localhost:5000/storage/info")
    print(f"curl -X POST http://localhost:5000/storage/rebuild")

def show_directory_structure():
    """Show the directory structure after implementing persistent storage"""
    
    print("\n📂 Directory Structure")
    print("=" * 50)
    
    structure = """
    ai-agent-ms/
    ├── app.py                     # Updated Flask app with persistent storage
    ├── setup.py                   # Updated with VectorDBPersistenceManager
    ├── files-db/                  # PDF files storage
    │   ├── *.pdf                  # Your uploaded PDF files
    └── vector_db_storage/         # NEW: Persistent storage directory
        ├── vector_db.pkl          # Pickled DataFrame with embeddings
        └── file_metadata.json     # File change tracking metadata
    """
    
    print(structure)
    
    print("Key Features:")
    print("• 🚀 Fast startup - loads existing embeddings instead of rebuilding")
    print("• 🔄 Smart updates - only processes new/changed files")
    print("• 💾 Persistent storage - survives app restarts")
    print("• 📊 Progress tracking - shows processing status")
    print("• 🎛️ Throttling - prevents API rate limit issues")
    print("• 🔍 File change detection - MD5 hash-based")

if __name__ == "__main__":
    try:
        test_storage_manager()
        test_api_endpoints()
        show_directory_structure()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"Start your Flask app with: python app.py")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()