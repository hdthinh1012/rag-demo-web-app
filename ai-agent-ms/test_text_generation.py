#!/usr/bin/env python3
"""
Test script for the new text generation system
This script demonstrates how to use the updated API that returns text instead of audio
"""

import requests
import json
import os
import time
from typing import Dict, Any

def test_text_generation_api():
    """Test the new text generation endpoint"""
    
    print("🧪 Testing Text Generation API")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   • RAG initialized: {health_data.get('rag_initialized')}")
            print(f"   • Documents indexed: {health_data.get('documents_indexed')}")
            print(f"   • PDF files count: {health_data.get('pdf_files_count')}")
            
            storage = health_data.get('storage', {})
            print(f"   • Storage available: {storage.get('persistent_storage_available')}")
            print(f"   • Indexed chunks: {storage.get('indexed_chunks', 0)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {str(e)}")
        return False
    
    # Test 2: Text generation without files
    print("\n2. Testing text generation with existing documents...")
    try:
        test_queries = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the most important findings mentioned?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            
            response = requests.post(
                f"{base_url}/generate-response",
                data={'query': query}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Response generated successfully")
                print(f"   • Response length: {len(result.get('response', ''))}")
                print(f"   • Context chunks used: {result.get('context_chunks')}")
                print(f"   • Timestamp: {result.get('timestamp')}")
                print(f"   • Response preview: {result.get('response', '')[:100]}...")
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
    except Exception as e:
        print(f"   ❌ Text generation error: {str(e)}")
    
    # Test 3: Test with file upload (if you have a test PDF)
    print("\n3. Testing text generation with file upload...")
    test_pdf_path = "./test_document.pdf"  # You can create a simple test PDF
    
    if os.path.exists(test_pdf_path):
        try:
            with open(test_pdf_path, 'rb') as pdf_file:
                files = {'files': pdf_file}
                data = {'query': 'What does this uploaded document contain?'}
                
                response = requests.post(
                    f"{base_url}/generate-response",
                    data=data,
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ File upload and processing successful")
                    print(f"   • Response: {result.get('response', '')[:150]}...")
                else:
                    print(f"   ❌ File upload failed: {response.status_code}")
                    print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ❌ File upload error: {str(e)}")
    else:
        print(f"   ℹ️ No test PDF found at {test_pdf_path} - skipping file upload test")
    
    # Test 4: Error handling
    print("\n4. Testing error handling...")
    try:
        # Test with empty query
        response = requests.post(f"{base_url}/generate-response", data={'query': ''})
        if response.status_code == 400:
            print("   ✅ Empty query handled correctly")
        else:
            print(f"   ⚠️ Unexpected response for empty query: {response.status_code}")
        
        # Test with missing query
        response = requests.post(f"{base_url}/generate-response", data={})
        if response.status_code == 400:
            print("   ✅ Missing query handled correctly")
        else:
            print(f"   ⚠️ Unexpected response for missing query: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error handling test failed: {str(e)}")
    
    return True

def compare_old_vs_new_api():
    """Compare the old audio API vs new text API"""
    
    print("\n📊 API Comparison: Audio vs Text")
    print("=" * 50)
    
    comparison = """
    OLD API (Audio Generation):
    ├── Endpoint: POST /generate-speech
    ├── Returns: Audio file (.wav)
    ├── Issues: Policy violations with Live API
    ├── Processing: Complex audio generation
    └── Usage: Download and play audio file
    
    NEW API (Text Generation):
    ├── Endpoint: POST /generate-response
    ├── Returns: JSON with text response
    ├── Benefits: No policy issues
    ├── Processing: Standard Gemini text API
    └── Usage: Direct text consumption
    """
    
    print(comparison)
    
    print("Example usage:")
    print("OLD: curl -X POST http://localhost:5000/generate-speech \\")
    print("      -F 'query=What is this about?' \\")
    print("      -F 'files=@document.pdf' \\")
    print("      --output response.wav")
    print()
    print("NEW: curl -X POST http://localhost:5000/generate-response \\")
    print("      -F 'query=What is this about?' \\")
    print("      -F 'files=@document.pdf'")
    print()
    print("Response format:")
    print(json.dumps({
        "response": "Based on the document content...",
        "query": "What is this about?",
        "timestamp": 1704067200,
        "context_chunks": 3
    }, indent=2))

def create_sample_test_requests():
    """Create sample requests for testing"""
    
    print("\n🔧 Sample Test Requests")
    print("=" * 50)
    
    samples = [
        {
            "name": "Basic Query",
            "method": "POST",
            "url": "http://localhost:5000/generate-response",
            "data": {"query": "What are the main topics discussed in the documents?"},
            "description": "Simple query without file upload"
        },
        {
            "name": "Query with File Upload",
            "method": "POST", 
            "url": "http://localhost:5000/generate-response",
            "data": {"query": "Summarize this document"},
            "files": "document.pdf",
            "description": "Query with new document upload"
        },
        {
            "name": "Health Check",
            "method": "GET",
            "url": "http://localhost:5000/health",
            "description": "Check system status and storage info"
        },
        {
            "name": "Storage Info",
            "method": "GET",
            "url": "http://localhost:5000/storage/info",
            "description": "Get detailed storage information"
        }
    ]
    
    for sample in samples:
        print(f"\n{sample['name']}:")
        print(f"  Description: {sample['description']}")
        print(f"  Method: {sample['method']}")
        print(f"  URL: {sample['url']}")
        if 'data' in sample:
            print(f"  Data: {sample['data']}")
        if 'files' in sample:
            print(f"  Files: {sample['files']}")

if __name__ == "__main__":
    print("🚀 Text Generation API Test Suite")
    print("Make sure your Flask app is running on http://localhost:5000")
    print()
    
    try:
        # Wait a moment for user to start the server if needed
        input("Press Enter when your Flask app is running, or Ctrl+C to exit...")
        
        success = test_text_generation_api()
        
        if success:
            compare_old_vs_new_api()
            create_sample_test_requests()
            print("\n🎉 Text generation system is working correctly!")
            print("Your RAG system now uses text responses instead of audio.")
        else:
            print("\n❌ Some tests failed. Please check your Flask app.")
            
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()