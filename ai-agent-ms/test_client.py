#!/usr/bin/env python3
"""
Simple test client for the AI Agent Microservice
"""

import requests
import os
import sys

def test_health_endpoint(base_url="http://localhost:5000"):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_generate_speech(query, pdf_files, base_url="http://localhost:5000", output_file="test_response.wav"):
    """Test the generate-speech endpoint"""
    try:
        # Prepare files for upload
        files_data = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                files_data.append(('files', (os.path.basename(pdf_file), open(pdf_file, 'rb'), 'application/pdf')))
            else:
                print(f"Warning: File {pdf_file} does not exist")
        
        if not files_data:
            print("No valid PDF files found")
            return False
        
        # Prepare form data
        data = {'query': query}
        
        print(f"Sending request with query: '{query}'")
        print(f"Files: {[f[1][0] for f in files_data]}")
        
        # Send request
        response = requests.post(
            f"{base_url}/generate-speech",
            data=data,
            files=files_data,
            timeout=120  # 2 minute timeout
        )
        
        # Close file handles
        for _, (_, file_handle, _) in files_data:
            file_handle.close()
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            # Save audio response
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Audio response saved to: {output_file}")
            print(f"Audio file size: {len(response.content)} bytes")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def list_files(base_url="http://localhost:5000"):
    """List uploaded files"""
    try:
        response = requests.get(f"{base_url}/files")
        print(f"Files List Status: {response.status_code}")
        if response.status_code == 200:
            files_info = response.json()
            print(f"Total files: {files_info['total_count']}")
            for file_info in files_info['files']:
                print(f"  - {file_info['filename']} ({file_info['size']} bytes)")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"List files failed: {e}")
        return False

def main():
    """Main test function"""
    if len(sys.argv) < 3:
        print("Usage: python test_client.py <query> <pdf_file1> [pdf_file2] ...")
        print("Example: python test_client.py 'What is this document about?' document.pdf")
        sys.exit(1)
    
    query = sys.argv[1]
    pdf_files = sys.argv[2:]
    
    print("=== AI Agent Microservice Test Client ===")
    print()
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    if test_health_endpoint():
        print("✓ Health check passed")
    else:
        print("✗ Health check failed")
        sys.exit(1)
    
    print()
    
    # Test generate speech endpoint
    print("2. Testing generate-speech endpoint...")
    if test_generate_speech(query, pdf_files):
        print("✓ Speech generation successful")
    else:
        print("✗ Speech generation failed")
        sys.exit(1)
    
    print()
    
    # List files
    print("3. Listing uploaded files...")
    if list_files():
        print("✓ Files listed successfully")
    else:
        print("✗ Failed to list files")
    
    print()
    print("=== All tests completed successfully ===")

if __name__ == "__main__":
    main()